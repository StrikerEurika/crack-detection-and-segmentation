from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from api.schemas import PredictResponse, ResultListResponse, ResultSummary
from api.services.result_store import result_store

router = APIRouter(prefix="/results", tags=["Results"])


@router.get("", response_model=ResultListResponse)
async def list_results(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    results = result_store.list_results(skip=skip, limit=limit)
    return ResultListResponse(total=result_store.total(), results=results)


@router.get("/{result_id}", response_model=PredictResponse)
async def get_result(result_id: str):
    record = result_store.get(result_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Result not found")

    base_url = f"/api/v1/results/{result_id}/visualization"
    data = record.to_dict()
    data["visualizations"] = {
        "overlay": f"{base_url}?type=overlay",
        "heatmap": f"{base_url}?type=heatmap",
        "mask": f"{base_url}?type=mask",
    }
    return PredictResponse(**data)


@router.get("/{result_id}/visualization")
async def get_visualization(result_id: str, type: str = Query("overlay")):
    valid_types = {"overlay", "heatmap", "mask", "error_analysis"}
    if type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid type. Must be one of: {valid_types}")

    path = result_store.get_visualization_path(result_id, type)
    if path is None:
        raise HTTPException(status_code=404, detail=f"{type} visualization not found for result {result_id}")

    return FileResponse(str(path), media_type="image/png")
