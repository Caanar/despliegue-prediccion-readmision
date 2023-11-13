from typing import Any, List, Optional

from pydantic import BaseModel
from model.processing.validation import DataInputSchema

# Esquema de los resultados de predicción
class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]

# Esquema para inputs múltiples
class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "num_lab_procedures": 0.57,
                        "num_medications": 0.4,
                        "number_inpatient": 1,
                        "time_in_hospital": 0.5,
                        "discharge_disposition_id": 0,
                        "number_diagnoses": 0.3
                    }
                ]
            }
        }
