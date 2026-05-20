import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import predict_flow

TS = "2006-10-27 08:00"
SITE = 3812


@pytest.mark.parametrize("model", ["lstm", "gru"])
def test_predict_positive_flow(model):
  if not os.path.exists(f"models/{model}_best.pth"):
    pytest.skip("model not trained")
  flow = predict_flow(SITE, TS, model=model)
  assert flow >= 0


def test_predict_invalid_site_raises():
  with pytest.raises(ValueError):
    predict_flow(99999, TS, model="lstm")
