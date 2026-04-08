from typing import Any, Dict, List


def select_best(results: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    if not results:
        raise ValueError("Cannot select best pipeline from an empty results list.")

    def sort_key(r: Dict[str, Any]):
        return (
            -r["metrics"][metric],
            r["spec"].complexity_score(),
            r["elapsed_sec"],
        )

    return sorted(results, key=sort_key)[0]
