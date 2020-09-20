import torch
import numpy as np
from math import sqrt

from functools import partial
from ignite.metrics import EpochMetric, Metric
import ignite.metrics
from scipy.special import softmax 
def outer_pairwise_distance(A, B=None):
    """
        Compute pairwise_distance of Tensors
            A (size(A) = n x d, where n - rows count, d - vector size) and
            B (size(A) = m x d, where m - rows count, d - vector size)
        return matrix C (size n x m), such as C_ij = distance(i-th row matrix A, j-th row matrix B)

        if only one Tensor was given, computer pairwise distance to itself (B = A)
    """

    if B is None: B = A

    max_size = 2 ** 26
    n = A.size(0)
    m = B.size(0)
    d = A.size(1)

    if n * m * d <= max_size or m == 1:

        return torch.pairwise_distance(
            A[:, None].expand(n, m, d).reshape((-1, d)),
            B.expand(n, m, d).reshape((-1, d))
        ).reshape((n, m))

    else:

        batch_size = max(1, max_size // (n * d))
        batch_results = []
        for i in range((m - 1) // batch_size + 1):
            id_left = i * batch_size
            id_rigth = min((i + 1) * batch_size, m)
            batch_results.append(outer_pairwise_distance(A, B[id_left:id_rigth]))

        return torch.cat(batch_results, dim=1)


def outer_cosine_similarity(A, B=None):
    """
        Compute cosine_similarity of Tensors
            A (size(A) = n x d, where n - rows count, d - vector size) and
            B (size(A) = m x d, where m - rows count, d - vector size)
        return matrix C (size n x m), such as C_ij = cosine_similarity(i-th row matrix A, j-th row matrix B)

        if only one Tensor was given, computer pairwise distance to itself (B = A)
    """

    if B is None: B = A

    max_size = 2 ** 32
    n = A.size(0)
    m = B.size(0)
    d = A.size(1)

    if n * m * d <= max_size or m == 1:

        A_norm = torch.div(A.transpose(0, 1), A.norm(dim=1)).transpose(0, 1)
        B_norm = torch.div(B.transpose(0, 1), B.norm(dim=1)).transpose(0, 1)
        return torch.mm(A_norm, B_norm.transpose(0, 1))

    else:

        batch_size = max(1, max_size // (n * d))
        batch_results = []
        for i in range((m - 1) // batch_size + 1):
            id_left = i * batch_size
            id_rigth = min((i + 1) * batch_size, m)
            batch_results.append(outer_cosine_similarity(A, B[id_left:id_rigth]))

        return torch.cat(batch_results, dim=1)


def metric_Recall_top_K(X, y, K, metric='cosine'):
    """
        calculate metric R@K
        X - tensor with size n x d, where n - number of examples, d - size of embedding vectors
        y - true labels
        N - count of closest examples, which we consider for recall calcualtion
        metric: 'cosine' / 'euclidean'.
            !!! 'euclidean' - to slow for datasets bigger than 100K rows
    """
    res = []

    n = X.size(0)
    d = X.size(1)
    max_size = 2 ** 32
    batch_size = max(1, max_size // (n*d))

    with torch.no_grad():

        for i in range(1 + (len(X) - 1) // batch_size):

            id_left = i*batch_size
            id_right = min((i+1)*batch_size, len(y))
            y_batch = y[id_left:id_right]

            if metric == 'cosine':
                pdist = -1 * outer_cosine_similarity(X, X[id_left:id_right])
            elif metric == 'euclidean':
                pdist = outer_pairwise_distance(X, X[id_left:id_right])
            else:
                raise AttributeError(f'wrong metric "{metric}"')

            values, indices = pdist.topk(K + 1, 0, largest=False)

            y_rep = y_batch.repeat(K, 1)
            res.append((y[indices[1:]] == y_rep).sum().item())

    return np.sum(res) / len(y) / K


class ignite_Recall_top_K(EpochMetric):

    def __init__(self, output_transform=lambda x: x, K=3, metric='cosine'):
        super(ignite_Recall_top_K, self).__init__(
            partial(metric_Recall_top_K, K = K, metric = metric), 
            output_transform=output_transform
        )


class BatchRecallTop(Metric):
    def __init__(self, k, metric='cosine'):
        super().__init__(output_transform=lambda x: x)

        self.num_value = 0.0
        self.denum_value = 0

        self.k = k
        self.metric = metric

    def reset(self):
        self.num_value = 0.0
        self.denum_value = 0

        super().reset()

    def update(self, output):
        x, y = output
        value = metric_Recall_top_K(x, y, self.k, self.metric)

        self.num_value += value
        self.denum_value += 1

    def compute(self):
        if self.denum_value == 0:
            return 0.0
        return self.num_value / self.denum_value


#custom class
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
class SpendPredictMetric(ignite.metrics.Metric):
  def __init__(self, ignored_class=None, output_transform=lambda x: x):
      self._relative_error = None
      super(SpendPredictMetric, self).__init__(output_transform=output_transform)

  @reinit__is_reduced
  def reset(self):
      self._relative_error = []
      super(SpendPredictMetric, self).reset()

  @reinit__is_reduced
  def update(self, output):
      y_pred, y = output
      delta = torch.abs(y_pred[:,0] - y[:,0])
      rel_delta = 100*delta / torch.max(y[:,0], torch.exp(y[:,0]-y[:,0]) )
      self._relative_error += [torch.mean(rel_delta).item()]

  @sync_all_reduce("_relative_error")
  def compute(self):
     if self._relative_error == 0:
       raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
     return sum(self._relative_error)/len(self._relative_error)

from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
class PercentPredictMetric(ignite.metrics.Metric):
   def __init__(self, ignored_class=None, output_transform=lambda x: x):
       self._relative_error = None
       self.softmax = torch.nn.Softmax(dim=1)
       super(PercentPredictMetric, self).__init__(output_transform=output_transform)

   @reinit__is_reduced
   def reset(self):
       self._relative_error = []
       super(PercentPredictMetric, self).reset()
   @reinit__is_reduced

   def numpy_estim(y_pred, y):
       soft_pred = softmax(y_pred[:,1:53], axis=1)
       delta = np.linalg.norm(soft_pred - y[:,1:53], ord=1,axis=1)
       rel_delta = 100*np.mean(delta)/2
       item_val =rel_delta.item()
       print(type(item_val))
       print(item_val)       
       return 0.0#item_val

   def update(self, output):
       y_pred, y = output
       soft_pred = self.softmax(y_pred[:,1:53])
       delta = torch.norm(soft_pred - y[:,1:53], dim=1)
       rel_delta = 100*torch.mean(delta)/sqrt(2)
       self._relative_error += [rel_delta.item()]

   @sync_all_reduce("_relative_error")
   def compute(self):
       if self._relative_error == 0:
          raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
       return sum(self._relative_error)/len(self._relative_error)                                                                                                                             

from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
class PercentR2Metric(ignite.metrics.Metric):

    def __init__(self, ignored_class=None, output_transform=lambda x: x):
       self._relative_error = None
       self.softmax = torch.nn.Softmax(dim=1)
       self.apriori_mean_list = [0.15042067,0.09637505,0.08652956,0.04166252,0.0446759 ,0.04421844, 0.04046275,0.01894829,0.01976015,0.03072916,0.02449278,0.01321937, 0.01667747,0.0056554 ,0.00944027,0.01454358,0.00691956,0.00495633, 0.00669245,0.00732218,0.0044744 ,0.00610466,0.00501702,0.00430312, 0.00450448,0.00470848,0.00479973,0.00559171,0.00532112,0.00511267, 0.0053236 ,0.00464357,0.00370333,0.00561579,0.00298665,0.00366132,
                                                              0.00423842,0.00283756,0.00100884,0.00258611,0.00287836,0.00324784, 0.00138557,0.00276069,0.0022538 ,0.00167298,0.00222069,0.0017486 ,  0.00148132,0.0018969 ,0.00256453,0.20564429]
                                    #[0.13848792, 0.1068176 ,0.11055607,0.04978109,0.0461238 ,0.04484921,0.03604756, 0.02170624,0.02311341,0.02929394,0.01710528,0.01494356,0.01408036,  0.00860207,0.00883866,0.01134359,0.00608305,0.00570947,0.00478716,  0.00506628,0.00462454,0.00450221,0.00477837,0.00360832,0.00401979,   0.00410604,0.00363408,0.00398468,0.00410963,0.00382853,0.00401247, 0.00352753,0.00336623,0.00319952,0.00349093,0.00318278,0.0031108 ,
                                    #        0.00304667,0.00134512,0.00248316,0.00260134,0.00219123,0.00191466, 0.00209553,0.00202027,0.00150444,0.00170138,0.00172567,0.0017353 , 0.00179628,0.00179591,0.20362025]
       self.apriori_mean_list = np.array(self.apriori_mean_list)
       super(PercentR2Metric, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._relative_error = []
        super(PercentR2Metric, self).reset()

    @reinit__is_reduced
    def update(self, output):
         y_pred, y = output
         soft_pred = self.softmax(y_pred[:,1:53])
         rss = torch.norm(soft_pred - y[:,1:53], dim=1)**2
         apriori_mean = np.tile(self.apriori_mean_list,(y_pred.shape[0],1))
         apriori_mean = torch.FloatTensor(apriori_mean).to(y.device)
         tss = torch.norm(soft_pred - apriori_mean, dim=1)**2
         r2 = 1 - rss/tss
         self._relative_error += [r2.mean().item()]
 
    @sync_all_reduce("_relative_error")
    def compute(self):
         if self._relative_error == 0:
            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
         return sum(self._relative_error)/len(self._relative_error)


from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
class MeanSumPredictMetric(ignite.metrics.Metric):

   def __init__(self, ignored_class=None, output_transform=lambda x: x):
      self._relative_error = None
      self.apriori_mean_list = [104.50246442561681, 24.13884404918004, 23.987986130468652, 25.99619133450599, 31.85280996717411, 48.45940328726219, 55.47242164432059, 30.274800540801422, 78.54245364057059, 33.584090641015216, 188.10486081552096, 36.28945151578021, 81.29908838191119, 13.36464317351852, 84.84840735088136, 151.9366333535613, 74.19875206712715, 33.440992543674014, 264.2680258351682, 37.409850273598, 202.8606972950071, 191.2214637202601, 40.240267318700866, 88.53218639904627, 126.69188207321002, 98.40358576861554, 133.64843240208026, 195.70574982471038, 190.9354436258683, 229.74390047798462, 167.13503214421857, 263.6275381853625, 61.9833114297043, 829.9443242145692, 70.43075570091183, 26.589740581794942, 120.50270399815835, 158.77900298555343, 42.00745367058008, 87.87754873721126, 88.79054102150519, 19.639214062447582, 517.5821513304905, 146.5522789130751, 149.49401859950868, 113.19210670119924, 61.46378188880782, 31.345534958521874, 78.6743993311771, 323.44568014813973, 346.0130976935031, 42.04684542519712]
                            #[56.45839298  20.11246728  17.12698011  17.62154946  24.66463394     31.60547122  38.43602175  18.01608807  18.87948067  41.40183747
                            # 123.46243218  20.81641073  51.39168202   7.17964714  38.15267886     77.96032626  50.25813267  15.39733476 108.48018051 137.78938747                       23.05712099  99.72804839  31.3945433   57.33839515  45.96509737  55.91229175  82.79980895 121.54191987  80.71129672 114.44696995 83.9339001   93.37572941  43.35426522 501.40469461  22.01595469  49.18496163  94.15643328  23.77541913  13.03441917  37.18790927  23.74937926 203.10674445  11.74796003  85.74747871  38.90868018  42.6732832  108.79567508  23.5477043   15.79474938  35.15901553  129.56101715  30.53026179]
      self.apriori_mean_list = np.array(self.apriori_mean_list)
      self.softmax = torch.nn.Softmax(dim=1)
      super(MeanSumPredictMetric, self).__init__(output_transform=output_transform)

      @reinit__is_reduced
      def reset(self):
         self._relative_error = []
         super(MeanSumPredictMetric, self).reset()

      @reinit__is_reduced
      def update(self, output):
         y_pred, y = output
         #for numerical stability, otherwise exp()**2 could be to big
         y_pred[y_pred > 15.0] = 15.0
         y_pred[:,:3] = output[0][:,:3]
         soft_pred = self.softmax(y_pred[:,1:53])
         soft_pred = torch.zeros(y[:,1:53].shape)
         soft_pred[y[:,1:53]>0] = 1
         soft_pred = soft_pred.to(y.device)
         rss = (torch.exp(y_pred[:,53:]) - torch.exp(y[:,53:]) )**2
         rss = soft_pred*rss # torch.max(y[:,53:], torch.exp(y[:,53:]-y[:,53:]) )i
         rss = rss.mean(axis=1)
         mean_apriori = np.tile(self.apriori_mean_list,(y_pred.shape[0],1)) 
         mean_apriori = torch.FloatTensor(mean_apriori)
         if y_pred.is_cuda:
             mean_apriori = mean_apriori.to(y_pred.get_device())
             tss = (torch.exp(y_pred[:,53:]) - mean_apriori)**2
             tss = soft_pred*tss
             tss = tss.mean(axis=1)
             r2 = 1 - rss/tss
             self._relative_error += [torch.mean(r2).item()]

      @sync_all_reduce("_relative_error")
      def compute(self):
        if self._relative_error == 0:
           raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return sum(self._relative_error)/len(self._relative_error)

