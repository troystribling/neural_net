# %%
%matplotlib inline
%reload_ext autoreload
%autoreload 2

%aimport os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
%aimport tensorflow

# %%
# Sample session with constants
session = tensorflow.Session()

deep_learning = tensorflow.constant('Deep Learning')
session.run(deep_learning)

a = tensorflow.constant(2)
b = tensorflow.constant(3)
multiply = tensorflow.multiply(a, b)
session.run(multiply)

# %%
# Common tensorflow varaiable initalizers
tensorflow.zeros([2,2], dtype=tensorflow.float64, name="Zeros")

# %%
# Sample session with variables
session = tensorflow.Session()
