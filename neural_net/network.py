# %%
%reload_ext autoreload
%autoreload 2

%aimport tempfile
%aimport os
%aimport numpy

# %%
w1 = 2.0 * numpy.random.rand(2, 2) - 1.0
w2 = 2.0 * numpy.random.rand(2, 2) - 1.0

print(w2)
