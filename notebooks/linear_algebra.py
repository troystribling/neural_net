# %%
%matplotlib inline
%reload_ext autoreload
%autoreload 2

%aimport numpy

# %%
# Column and row vectors

row_vector = numpy.array([2, 3])
column_vector = numpy.array(row_vector, ndmin=2).T

row_vector.shape
column_vector.shape
numpy.transpose(column_vector).shape

row_vector * column_vector
column_vector * row_vector
row_vector * row_vector
column_vector * column_vector
numpy.dot(row_vector, column_vector)
numpy.dot(row_vector, row_vector)
numpy.dot(column_vector, numpy.transpose(column_vector))
numpy.dot(column_vector, numpy.array([4, 2], ndmin=2))

# %%
# 2 dimenstianl array operations
square_matrix = numpy.array([[1, 2],
                             [3, 4]])
square_matrix_transpose = square_matrix.T

numpy.transpose(square_matrix)
square_matrix * square_matrix_transpose
square_matrix_transpose * square_matrix

numpy.dot(square_matrix_transpose, square_matrix)
numpy.dot(square_matrix, square_matrix_transpose)

row_vector = numpy.array([2, 3])
column_vector = numpy.array(row_vector, ndmin=2).T
numpy.transpose(column_vector)

square_matrix * row_vector
row_vector * square_matrix

square_matrix * column_vector
column_vector * square_matrix

numpy.dot(square_matrix, row_vector)
numpy.dot(row_vector, square_matrix)

numpy.dot(square_matrix, column_vector)
numpy.dot(numpy.transpose(column_vector), square_matrix)
