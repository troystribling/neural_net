# %%
%matplotlib inline
%reload_ext autoreload
%autoreload 2

%aimport numpy

# %%
# Column and row vectors
vector = numpy.array([2, 3])
row_vector = vector.reshape((1,2))
column_vector = row_vector.T
numpy.transpose(row_vector)
numpy.array(vector, ndmin=2)

vector.shape
row_vector.shape
column_vector.shape
column_vector.T.shape

vector * row_vector

vector * column_vector

column_vector * vector

row_vector * column_vector

column_vector * row_vector

vector * vector

row_vector * row_vector

column_vector * column_vector

numpy.dot(vector, column_vector)
numpy.dot(row_vector, column_vector)
numpy.dot(column_vector, row_vector)


# %%
# 2 dimensional array operations
square_matrix = numpy.array([[1, 2],
                             [3, 4]])
square_matrix_transpose = square_matrix.T

numpy.transpose(square_matrix)
square_matrix * square_matrix_transpose
square_matrix_transpose * square_matrix

numpy.dot(square_matrix_transpose, square_matrix)
numpy.dot(square_matrix, square_matrix_transpose)

row_vector = numpy.array([[2, 3]])
column_vector = row_vector.T
numpy.transpose(column_vector)

square_matrix * row_vector
row_vector * square_matrix

square_matrix * column_vector
column_vector * square_matrix

numpy.dot(row_vector, square_matrix)

numpy.dot(square_matrix, column_vector)
