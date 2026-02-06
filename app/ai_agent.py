import json
import os
import re
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file (if present),
# so OPENAI_API_KEY can be configured without hardcoding.
load_dotenv()

ProblemType = Literal["classification", "regression"]


_ALLOWED_MODELS: Dict[ProblemType, List[str]] = {
    "classification": ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"],
    "regression": ["Linear Regression", "Random Forest Regressor", "Gradient Boosting", "SVR"],
}


def _fallback_decision(df: pd.DataFrame) -> Dict[str, Any]:
    """Heuristic fallback if the LLM is unavailable or returns invalid JSON."""
    target_col = df.columns[-1] if len(df.columns) else ""
    problem_type: ProblemType = "classification"

    if target_col:
        y = df[target_col]
        # If numeric with many unique values, treat as regression.
        if pd.api.types.is_numeric_dtype(y):
            uniq = int(y.nunique(dropna=True))
            if uniq > 20:
                problem_type = "regression"

    models = _ALLOWED_MODELS[problem_type]
    return {
        "target_column": target_col,
        "problem_type": problem_type,
        "models": models,
        "reasoning": "Fallback heuristic: used last column as target; inferred task from target dtype/unique values.",
    }


def _safe_json_loads(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_json_object(text: str) -> Optional[dict]:
    """Extract the first JSON object from a string."""
    # Greedy match for the largest object; then parse.
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    return _safe_json_loads(m.group(0))


def _normalize_keys(raw: dict) -> dict:
    """Normalize dict keys: strip all whitespace/quotes/backslashes, match expected names."""
    out = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        # Remove all whitespace, quotes, backslashes so "\n \"target_column\"" -> "target_column"
        key_clean = re.sub(r'[\s"\'\\]+', '', k).lower().replace("-", "_")
        if not key_clean:
            continue
        if key_clean in ("target_column", "targetcolumn"):
            out["target_column"] = v
        elif key_clean in ("problem_type", "problemtype"):
            out["problem_type"] = v
        elif key_clean == "models":
            out["models"] = v
        elif key_clean == "reasoning":
            out["reasoning"] = v
    return out


def _normalize_decision(raw: Any, df: pd.DataFrame) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return _fallback_decision(df)

    raw = _normalize_keys(raw)

    target = str(raw.get("target_column", "")).strip()
    if target not in df.columns:
        target = df.columns[-1] if len(df.columns) else target

    pt = str(raw.get("problem_type", "")).strip().lower()
    problem_type: ProblemType = "classification" if pt != "regression" else "regression"

    models_in = raw.get("models", [])
    if not isinstance(models_in, list):
        models_in = []
    models = [str(m).strip() for m in models_in if str(m).strip()]

    # Keep only supported model names (so train pipeline can honor them).
    allowed = set(_ALLOWED_MODELS[problem_type])
    models = [m for m in models if m in allowed]
    if not models:
        models = _ALLOWED_MODELS[problem_type]

    reasoning = str(raw.get("reasoning", "")).strip() or "No reasoning provided."

    return {
        "target_column": target,
        "problem_type": problem_type,
        "models": models,
        "reasoning": reasoning,
    }


def analyze_dataset(csv_path: str) -> Dict[str, Any]:
    """
    Analyze a CSV dataset and return an AutoML decision.

    Output JSON (as Python dict) strictly:
    {
      "target_column": "...",
      "problem_type": "classification/regression",
      "models": ["Model1", "Model2"],
      "reasoning": "Short explanation..."
    }
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {
            "target_column": "",
            "problem_type": "classification",
            "models": _ALLOWED_MODELS["classification"],
            "reasoning": "Could not load dataset.",
        }

    if df.shape[1] == 0:
        return {
            "target_column": "",
            "problem_type": "classification",
            "models": _ALLOWED_MODELS["classification"],
            "reasoning": "Dataset has no columns.",
        }

    # One big try/except: any error (LLM, JSON, keys) -> use heuristic fallback
    try:
        col_summaries = []
        for col in df.columns[:60]:
            s = df[col]
            col_summaries.append({
                "name": str(col),
                "dtype": str(s.dtype),
                "unique": int(s.nunique(dropna=True)),
                "missing_pct": float(s.isna().mean() * 100.0),
                "example_values": [str(v) for v in s.dropna().astype(str).head(3).tolist()],
            })

        prompt = PromptTemplate(
            input_variables=["dataset_shape", "columns_json", "allowed_models_json"],
            template=(
                "You are an AutoML AI Agent.\n"
                "You must decide target column, problem type, and which models to train.\n\n"
                "Dataset shape: {dataset_shape}\n"
                "Columns summary (JSON array):\n{columns_json}\n\n"
                "You can ONLY choose model names from this allowed list (JSON):\n{allowed_models_json}\n\n"
                "Rules:\n"
                "- Pick exactly one target_column from the given columns.\n"
                "- problem_type must be exactly \"classification\" or \"regression\".\n"
                "- models must be a non-empty array of allowed model names.\n"
                "- reasoning must be 1-3 short sentences.\n\n"
                "Respond with ONLY a JSON object, no markdown, no extra text.\n"
                "Required JSON format:\n"
                "{\n"
                "  \"target_column\": \"...\",\n"
                "  \"problem_type\": \"classification/regression\",\n"
                "  \"models\": [\"Model1\", \"Model2\"],\n"
                "  \"reasoning\": \"Short explanation...\"\n"
                "}\n"
            ),
        )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return _fallback_decision(df)

        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
        rendered = prompt.format(
            dataset_shape=str(df.shape),
            columns_json=json.dumps(col_summaries, ensure_ascii=False, default=str),
            allowed_models_json=json.dumps(_ALLOWED_MODELS, ensure_ascii=False),
        )

        invoke_fn = getattr(llm, "invoke", None) or getattr(llm, "predict", None)
        resp = invoke_fn(rendered)
        text = getattr(resp, "content", resp)
        if not isinstance(text, str):
            text = str(text) if text is not None else "{}"
        text = (text or "").strip()

        parsed = _safe_json_loads(text) or _extract_json_object(text)
        return _normalize_decision(parsed, df)
    except Exception:
        return _fallback_decision(df)
