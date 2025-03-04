import math

import numpy 

from model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_no_predictions = 20353

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    print(predictions[0])
    #assert isinstance(predictions[0], numpy.floating)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions

