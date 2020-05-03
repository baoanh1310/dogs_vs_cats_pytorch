from lib import *

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

WIDTH = 150
HEIGHT = 150

BATCH_SIZE = 20
NUM_EPOCHS = 100
MEAN = 1./255
STD = 1.0

save_path = './dogs_cats.pth'