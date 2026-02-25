import model.lib_teacher.tresnet_v2
import model.mobilemamba.mobilemamba
import model.starnet.starnet
import model.starnet.starnet_shuffle_ffn
import model.starnet.starnet_new
#import model.starnet.starnet_new_shuffle
#import model.starnet.starnt_64_new_192
import model.fasternet.fasternet 
import model.restarnet.starnet 
import model.efficientvit.build
import model.WTNet.build
import model.starnet.starnet_new_mhsa
# import model.WCANet.WCANet
import model.cspdarknet.yolo11
import model.cspconvnext.cspconvnext
import model.cspconvnext.weconvnext
import model.WCANet.cspwcanet

import model.starnet.starnet_new_unlearn
# import model.WCANet_windows.build
#import model.starnet.starnet_new_attn
# import model.mobilenetv4.build_mobilenet_v4
#import model.starnet.starnet_new2


from timm.models._registry import _model_entrypoints
from . import MODEL

for timm_name, timm_fn in _model_entrypoints.items():
	MODEL.register_module(timm_fn, f'timm_{timm_name}')

if __name__ == '__main__':
	print()
	