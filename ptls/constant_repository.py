from enum import Enum
import torch

class TorchDataTypeConstant(Enum):
    TORCH_FLOAT16 = torch.float16
    TORCH_BFLOAT16 = torch.bfloat16
    TORCH_FLOAT32 = torch.float32
    TORCH_FLOAT64 = torch.float64
    TORCH_INT8 = torch.int8
    TORCH_INT16 = torch.int16
    TORCH_INT32 = torch.int32
    TORCH_INT64 = torch.int64
    TORCH_BOOL = torch.bool



TORCH_FLOAT16 = TorchDataTypeConstant.TORCH_FLOAT16.value
TORCH_BFLOAT16 = TorchDataTypeConstant.TORCH_BFLOAT16.value
TORCH_FLOAT32 = TorchDataTypeConstant.TORCH_FLOAT32.value
TORCH_FLOAT64 = TorchDataTypeConstant.TORCH_FLOAT64.value
TORCH_INT8 = TorchDataTypeConstant.TORCH_INT16.value
TORCH_INT16 = TorchDataTypeConstant.TORCH_INT16.value
TORCH_INT32 = TorchDataTypeConstant.TORCH_INT32.value
TORCH_INT64 = TorchDataTypeConstant.TORCH_INT64.value
TORCH_BOOL = TorchDataTypeConstant.TORCH_BOOL.value

TORCH_EMB_DTYPE = TORCH_FLOAT32
TORCH_DATETIME_DTYPE = TORCH_INT32
TORCH_GROUP_DTYPE = TORCH_INT16