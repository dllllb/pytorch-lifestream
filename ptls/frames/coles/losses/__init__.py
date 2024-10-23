from .contrastive_loss import ContrastiveLoss, MultiContrastiveLoss, IMContrastiveLoss
from .club_loss import CLUBLoss
from .margin_loss import MarginLoss
from .binomial_deviance_loss import BinomialDevianceLoss
from .barlow_twins_loss import BarlowTwinsLoss
from .vicreg_loss import VicregLoss
from .cluster_loss import ClusterLoss, ClusterAndContrastive

from .triplet_loss import TripletLoss

from .histogram_loss import HistogramLoss
from .centroid_loss import CentroidLoss, CentroidSoftmaxLoss, CentroidSoftmaxMemoryLoss

from .softmax_loss import SoftmaxLoss
