import imp
from models.icarl import iCaRL
from models.end2end import End2End
from models.dr import DR
from models.ucir import UCIR
from models.bic import BiC
from models.lwm import LwM
from models.podnet import PODNet
from models.icarl_adv import iCaRL_adv
from models.twobn_cl import twobn_cl
from models.twobn_cl_inverse import twobn_cl_inverse
from models.twobn_eeil_inverse import twobn_eeil_inverse
from models.multibn_cl import multibn_cl
from models.twobn_bic import twobn_bic
from models.twobn_bic_inverse import twobn_bic_inverse
from models.twobn_cl_inverse_v2 import twobn_cl_inverse_v2
from models.twobn_cl_inverse_mixup import twobn_cl_inverse_mixup
from models.icarl_consistency_regularization import icarl_regularization
from models.icarl_regularization_v2 import icarl_regularization_v2
from models.icarl_regularization_v3 import icarl_regularization_v3
from models.icarl_regularization_v4 import icarl_regularization_v4
from models.icarl_regularization_v5 import icarl_regularization_v5
from models.icarl_regularization_v6 import icarl_regularization_v6
from models.icarl_regularization_v7 import icarl_regularization_v7
from models.icarl_regularization_v8 import icarl_regularization_v8
from models.icarl_regularization_v9 import icarl_regularization_v9
from models.icarl_regularization_v10 import icarl_regularization_v10
from models.multi_bn import multi_bn
from models.icarl_generator_fixed import icarl_generator_fixed
from models.multi_bn_pretrained import multi_bn_pretrained
from models.multi_bn_pretrained_kw import multi_bn_pretrained_kw
from models.single_bn_pretrain import single_bn_pretrained
from models.multi_bn_mixup import multi_bn_mixup


def get_model(model_name, args):
    name = model_name.lower()
    if name == 'icarl':
        return iCaRL(args)
    elif name == 'end2end':
        return End2End(args)
    elif name == 'dr':
        return DR(args)
    elif name == 'ucir':
        return UCIR(args)
    elif name == 'bic':
        return BiC(args)
    elif name == 'lwm':
        return LwM(args)
    elif name == 'podnet':
        return PODNet(args)
    elif name == 'icarl_adv':
        return iCaRL_adv(args)
    elif name == 'twobn_cl':
        return twobn_cl(args)
    elif name == 'twobn_cl_inverse':
        return twobn_cl_inverse(args)
    elif name == 'twobn_cl_inverse_v2':
        return twobn_cl_inverse_v2(args)
    elif name == 'twobn_eeil_inverse':
        return twobn_eeil_inverse(args)
    elif name == 'multibn_cl':
        return multibn_cl(args)
    elif name == 'twobn_bic':
        return twobn_bic(args)
    elif name == 'twobn_bic_inverse':
        return twobn_bic_inverse(args)
    elif name == 'twobn_cl_inverse_mixup':
        return twobn_cl_inverse_mixup(args)
    elif name == 'icarl_regularization':
        return icarl_regularization(args)
    elif name == 'icarl_regularization_v2':
        return icarl_regularization_v2(args)
    elif name == 'icarl_regularization_v3':
        return icarl_regularization_v3(args)
    elif name == 'icarl_regularization_v4':
        return icarl_regularization_v4(args)
    elif name == 'icarl_regularization_v5':
        return icarl_regularization_v5(args)
    elif name == 'icarl_regularization_v6':
        return icarl_regularization_v6(args)
    elif name == 'icarl_regularization_v7':
        return icarl_regularization_v7(args)
    elif name == 'icarl_regularization_v8':
        return icarl_regularization_v8(args)
    elif name == 'icarl_regularization_v9':
        return icarl_regularization_v9(args)
    elif name == 'icarl_regularization_v10':
        return icarl_regularization_v10(args)
    elif name == 'multi_bn':
        return multi_bn(args)
    elif name == "icarl_generator_fixed":
        return icarl_generator_fixed(args)
    elif name == "multi_bn_pretrained":
        return multi_bn_pretrained(args)
    elif name == "multi_bn_pretrained_kw":
        return multi_bn_pretrained_kw(args)
    elif name == "single_bn_pretrained":
        return single_bn_pretrained(args)
    elif name == "multi_bn_mixup":
        return multi_bn_mixup(args)