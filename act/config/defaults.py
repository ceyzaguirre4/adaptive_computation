from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# MODEL definition
# -----------------------------------------------------------------------------
_C = CN()

_C.MODEL = CN()
_C.MODEL.RNN_BASELINE = False
_C.MODEL.GRU = False
_C.MODEL.LSTM = False
_C.MODEL.MAX_ITER = 20
_C.MODEL.TASK = "parity"
_C.MODEL.DEVICE = "cuda"
_C.MODEL.CONTROLLER = CN()
_C.MODEL.CONTROLLER.HIDDEN_SIZE = 128
# _C.MODEL.SEED = 1


# -----------------------------------------------------------------------------
# ACT
# -----------------------------------------------------------------------------
_C.ACT = CN()
_C.ACT.HALT_TRAIN = True
_C.ACT.HALT_TEST = True
_C.ACT.LINEAR_PENALTY = True
_C.ACT.PENALTY_COEF = 1e-3
_C.ACT.MIN_PENALTY = 0.0
_C.ACT.BASELINE = CN()
_C.ACT.BASELINE.EPSILON = 1e-2


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.DIM = 64
_C.INPUT.SEQ_LEN = 1


# -----------------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------------
_C.OUTPUT = CN()
_C.OUTPUT.DIM = 1


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 1
_C.DATALOADER.BATCH_SIZE = 128
_C.DATALOADER.TRAIN_SAMPLES = 25000
_C.DATALOADER.TRAIN_TEST_PROPORTION = 2e-1


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.EPOCHS = 40000
_C.SOLVER.LR = 1e-4
_C.SOLVER.GRAD_CLIP = 0.0
_C.SOLVER.USE_SCHEDULER = False
_C.SOLVER.NO_GATE_TRAIN = False


# ---------------------------------------------------------------------------- #
# weight saving/loading options
# ---------------------------------------------------------------------------- #
_C.SAVE = True
_C.SAVE_PATH = "saved/checkpoint.pth.tar"
_C.SAVE_INTERVAL = 10
_C.LOAD = False
_C.LOAD_BEST = True
_C.LOAD_PATH = "saved/checkpoint.pth.tar"
