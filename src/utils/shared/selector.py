from typing import Any, Dict, List


def ranking_score(result: Dict[str, Any], metric: str) -> float:
    if "normalized_score" in result:
        return float(result["normalized_score"])
    if "final_score" in result:
        return float(result["final_score"])
    return float(result["metrics"][metric])


def select_best(results: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    if not results:
        raise ValueError("Cannot select best pipeline from an empty results list.")

    def sort_key(r: Dict[str, Any]):
        return (
            -ranking_score(r, metric),
            r["spec"].complexity_score(),
            r["elapsed_sec"],
        )

    return sorted(results, key=sort_key)[0]
