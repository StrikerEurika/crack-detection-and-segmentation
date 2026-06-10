from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from api.schemas import PredictResponse, ResultListResponse, ResultSummary
from api.services.result_store import result_store

router = APIRouter(prefix="/results", tags=["Results"])


@router.get(
    "",
    response_model=ResultListResponse,
    summary="List Inference Results",
    description="Returns a paginated list of previously run inference results, including detection status, confidence scores, and timestamps. Results are ordered from newest to oldest.",
    response_description="Paginated list of inference results",
)
async def list_results(
    skip: int = Query(0, ge=0, description="Number of results to skip for pagination"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results to return (max 200)"),
):
    results = result_store.list_results(skip=skip, limit=limit)
    return ResultListResponse(total=result_store.total(), results=results)


@router.get(
    "/{result_id}",
    response_model=PredictResponse,
    summary="Get Result by ID",
    description="Retrieves the full inference result for a given result_id, including crack detection status, confidence, crack dimensions, and visualization URLs. Returns 404 if the result does not exist.",
    response_description="Full inference result details",
    responses={404: {"description": "Result not found"}},
)
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


@router.get(
    "/{result_id}/visualization",
    summary="Get Visualization Image",
    description="Retrieves a PNG visualization image for a given result. Supports overlay (prediction on original), heatmap (confidence map), mask (binary prediction), and error_analysis (available when ground-truth was provided).",
    response_description="PNG visualization image",
    responses={
        400: {"description": "Invalid visualization type"},
        404: {"description": "Visualization not found for this result"},
    },
)
async def get_visualization(
    result_id: str,
    type: str = Query("overlay", description='Visualization type: "overlay", "heatmap", "mask", or "error_analysis"'),
):
    valid_types = {"overlay", "heatmap", "mask", "error_analysis"}
    if type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid type. Must be one of: {valid_types}")

    path = result_store.get_visualization_path(result_id, type)
    if path is None:
        raise HTTPException(status_code=404, detail=f"{type} visualization not found for result {result_id}")

    return FileResponse(str(path), media_type="image/png")
