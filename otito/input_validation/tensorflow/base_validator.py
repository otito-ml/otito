from pydantic import BaseModel
import tensorflow as tf


class BaseTensorflowValidator(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        kwargs["y_predicted"] = tf.reshape(kwargs["y_predicted"], [-1])
        super().__init__(**kwargs)
