from keras.engine import Layer 
from keras import backend as K
class Sum(Layer):
    '''
    sum wordemb with mask supporting
    '''
    input_ndim = 2

    def __init__(self, input_dim,input_length=None,
                  ave=False,**kwargs):
        self.input_dim = input_dim
        self.input_length = input_length
        self.output_dim = input_dim
        self.supports_masking = True
        self.ave = ave
        kwargs['input_shape'] = (self.input_length,self.input_dim)
        super(Sum, self).__init__(**kwargs)

    def compute_mask(self, input, mask=None):
        return None

    def build(self, input_shape=None):
        pass

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],self.output_dim)

    def call(self, x, mask=None):
        num = self.input_length
        if mask:
            num = mask.sum(-1,keepdims=True)
            num = K.cast(num,K.floatx())
            mask = K.expand_dims(mask,-1)
            _x = x*K.cast(mask,K.floatx())
        else:
            _x = x
        if not self.ave:
            num = 1
        return K.sum(_x,1)/num

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'output_dim': self.output_dim,
                  'input_length': self.input_length,
                  'mask_zero': self.mask_zero,
                  'ave': slef.ave,
                  }
        base_config = super(Sum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



if __name__=="__main__":
  from keras.models import Sequential
  from keras.layers import Embedding,Dense
  import numpy as np
  def test1():
    model = Sequential()
    model.add(Embedding(100,50,input_length=10,mask_zero=True))
    model.add(Sum(50,ave=True))
    model.compile(optimizer='sgd', loss='mse')
    a = model.predict(np.array([range(10)]))
    w = model.get_weights()[0]
    b = w[1:10,:].mean(0)
    if abs((a-b).sum())<1e-8:
      print("Behave as expectation")
    else:
      print("Something wrong")
  def test2():
      model = Sequential()
      model.add(Embedding(100,50,input_length=10,mask_zero=True))
      model.add(Sum(50,ave=True))
      model.add(Dense(10))
      model.compile(optimizer='sgd', loss='mse')
      a = model.predict(np.array([range(10)]))
      print(a)
  test1()
  test2()
