"""
Copyright (C) 2018 - Rickard Sjoegren
"""
class _gsd:
    
    def __init__(
        self,
        levels:Iterable[int],
        reduction:int,
        progress_bar:callable = None,
    ) -> None:

        try:
            levels = [*map(int, levels)]
        except (ValueError, TypeError):
            'levels has to be sequence of integers'

        reduction = int(reduction)
        assert reduction > 1, 'reduction has to be integer larger than 1'
        self.levels = levels
        self.reduction = reduction
        self._intype_dict = {np.iinfo(d).max:d for d in (np.int8, np.int16, np.int32, np.int64)}
        self.final_partitions = None
        self._orthogonals_generated = False
        self.progress = progress_bar


    def get_optimal_intype(self, max_num:int) -> np.dtype: 
        return self._intype_dict[min(size for size in self._intype_dict if max_num <= size)]        


    def _make_partitions(
        self,
        levels:List[int], 
        reduction:int,
    ) -> List[List[list]]:

        partitions = []
        partitions_len = []
        max_len = 0
        for partition_i in range(1, reduction + 1):
            partition = []
            partitioncount = []
            for num_levels in levels:
                part = []
                partcount = 0
                for level_i in range(1, num_levels):
                    index = partition_i + (level_i - 1) * reduction
                    if index <= num_levels:
                        part.append(index)
                        partcount += 1
                partition.append(part)
                partitioncount.append(partcount)
                if partcount > max_len:
                    max_len = partcount
            partitions.append(partition)
            partitions_len.append(partitioncount)

        try:
            self.partitions = np.array(
                partitions, 
                dtype = self.get_optimal_intype(num_levels),
            )
#             self._partitions_len = np.full(
#                 self.partitions.shape[:-1],
#                 self.partitions.shape[-1],
#             )
        except ValueError:
            self.partitions = np.array(partitions, dtype = object)
#             nplen = np.vectorize(len)
#             self._partitions_len = nplen(partitions)

        self._partitions_len = np.array(
            partitions_len, 
            dtype = self.get_optimal_intype(max_len),
        )

        return self.partitions


    def _make_latin_square(
        self,
        reduction:int,
    ) -> np.ndarray:

        rang = np.arange(0,reduction,dtype = self.get_optimal_intype(2*reduction))
        self.latin_square = np.repeat(rang.reshape(1,-1),reduction,axis=0)
        self.latin_square += rang.reshape(-1,1)
        self.latin_square[self.latin_square >= reduction] -= reduction
        return self.latin_square


    def _make_orthogonal_arrays(
        self,
        latin_square:np.ndarray,
        num_factors:int,
    ) -> np.ndarray:

        p = len(latin_square)
        first_row = latin_square[0]
        A_matrices = first_row.reshape(-1,1,1)
        while A_matrices[0].shape[1] < num_factors:
            A_matrices = np.array(
                [np.vstack([np.hstack([np.repeat(constant, len(other_A))[:, np.newaxis], other_A]) 
                for constant,other_A in zip(first_row, A_matrices[latin_square[n]])])
                for n in range(A_matrices.shape[0])],
                dtype = self.get_optimal_intype(latin_square.max()),
            )
        return A_matrices


    def _generate_orthogonals(
        self,
    ) -> None:

        if not self._orthogonals_generated:        
            self.num_levels = len(self.levels)
            self.partitions = self._make_partitions(self.levels, self.reduction)
            self.latin_square = self._make_latin_square(self.reduction)
            self.orthogonal_arrays = self._make_orthogonal_arrays(self.latin_square, self.num_levels)
            self._orthogonals_generated = True


    def _index_partitions(
        self,
        partitions:np.ndarray,
        orthogonal_array:np.ndarray,
        iter_wrapper:callable = None,
    ) -> Iterable:

        assert (len(partitions) == orthogonal_array.max() + 1 and not orthogonal_array.min()), \
        'Orthogonal array indexing does not match partition structure'
        col_indexer = np.arange(orthogonal_array.shape[1])
        self._partitions_iter = partitions[orthogonal_array, col_indexer]
        self._partitions_len_i = self._partitions_len[orthogonal_array, col_indexer]
        self._partition_bool = np.bool_(self._partitions_iter).all(axis=-1)
        if partitions.dtype != object:
            self._partition_bool = self._partition_bool.all(axis=-1)
        self._partitions_iter = self._partitions_iter[self._partition_bool]
        self._partitions_len_i = self._partitions_len_i[self._partition_bool]
        if iter_wrapper:
            self._partitions_iter = iter_wrapper(self._partitions_iter)
        return self._partitions_iter, self._partitions_len_i


    def _map_partitions_to_design(
        self,
        partitions_iterable:Iterable,
        num_levels:int,
        partitions_len:np.ndarray,
    ) -> np.ndarray:

        num_experiments = partitions_len.prod(axis=-1).astype(np.int64).sum()
        num_values = num_experiments * np.int64(num_levels)

        if num_values > 100_000:
            print('Parallel processing engaged')
            self.final_experiments = iters.chain.from_iterable(
                Parallel(n_jobs=-1)
                (delayed(iters.product)(*row) for row in partitions_iterable)
            )
        else:
            self.final_experiments = iters.chain.from_iterable(
                iters.starmap(iters.product, partitions_iterable)
            )

        try:
            self.final_experiments = np.fromiter(
                iters.chain.from_iterable(self.final_experiments),
                dtype = self.get_optimal_intype(num_levels),
                count = num_values
            ).reshape(num_experiments, num_levels) - 1
            return self.final_experiments
        except ValueError:
            raise ValueError('reduction too large compared to factor levels')
        except MemoryError:
            raise MemoryError('Experimental matrix too large to be generated in a structured \
            manner. Please use another method.')


    def generate(
        self,
        n_return_designs:int = 1,
    ) -> np.ndarray:

        self.n_return_designs = int(n_return_designs)
        assert self.n_return_designs, 'n_return_designs has to be a positive integer'
        if not self._orthogonals_generated:
            self._generate_orthogonals()

        if self.n_return_designs > 1:
            self.return_designs = []
            for n,orthogonal_array in enumerate(self.orthogonal_arrays[:self.n_return_designs],1):
                print(f'calculating design {n}')
                partitions_iter, partitions_len = self._index_partitions(
                    partitions = self.partitions,
                    orthogonal_array = orthogonal_array,
                    iter_wrapper = self.progress,
                )
                self.n_return_designs.append(self._map_partitions_to_design(
                    partitions_iterable = partitions_iter, 
                    num_levels = self.num_levels,
                    partitions_len = partitions_len,
                ))

        else:
            partitions_iter, partitions_len = self._index_partitions(
                partitions = self.partitions,
                orthogonal_array = self.orthogonal_arrays[0],
                iter_wrapper = self.progress,
            )
            return self._map_partitions_to_design(
                partitions_iterable = partitions_iter, 
                num_levels = self.num_levels,
                partitions_len = partitions_len,
            )


def gsd(levels, reduction, n=1):
    """
    Create a Generalized Subset Design (GSD).

    Parameters
    ----------
    levels : array-like
        Number of factor levels per factor in design.
    reduction : int
        Reduction factor (bigger than 1). Larger `reduction` means fewer
        experiments in the design and more possible complementary designs.
    n : int
        Number of complementary GSD-designs (default 1). The complementary
        designs are balanced analogous to fold-over in two-level fractional
        factorial designs.

    Returns
    -------
    H : 2d-array | list of 2d-arrays
        `n` m-by-k matrices where k is the number of factors (equal
        to the length of `factor_levels`. The number of rows, m, will
        be approximately equal to the grand product of the factor levels
        divided by `reduction`.

    Raises
    ------
    ValueError
        If input is valid or if design construction fails. Design can fail
        if `reduction` is too large compared to values of `levels`.

    Notes
    -----
    The Generalized Subset Design (GSD) [1]_ or generalized factorial design is
    a generalization of traditional fractional factorial designs to problems
    where factors can have more than two levels.

    In many application problems factors can have categorical or quantitative
    factors on more than two levels. Previous reduced designs have not been
    able to deal with such types of problems. Full multi-level factorial
    designs can handle such problems but are however not economical regarding
    the number of experiments.

    Note for commercial users, the application of GSD to testing of product
    characteristics in a processing facility is patented [2]_

    Examples
    --------
    An example with three factors using three, four and
    six levels respectively reduced with a factor 4 ::

        >>> gsd([3, 4, 6], 4)
        array([[0, 0, 0],
               [0, 0, 4],
               [0, 1, 1],
               [0, 1, 5],
               [0, 2, 2],
               [0, 3, 3],
               [1, 0, 1],
               [1, 0, 5],
               [1, 1, 2],
               [1, 2, 3],
               [1, 3, 0],
               [1, 3, 4],
               [2, 0, 2],
               [2, 1, 3],
               [2, 2, 0],
               [2, 2, 4],
               [2, 3, 1],
               [2, 3, 5]])

    Two complementary designs with two factors using three and
    four levels reduced with a factor 2 ::

        >>> gsd([3, 4], 2, n=2)[0]
        array([[0, 0],
               [0, 2],
               [2, 0],
               [2, 2],
               [1, 1],
               [1, 3]])
        >>> gsd([3, 4], 2, n=2)[1]
        array([[0, 1],
               [0, 3],
               [2, 1],
               [2, 3],
               [1, 0],
               [1, 2]])

    If design fails ValueError is raised ::

        >>> gsd([2, 3], 5)
        Traceback (most recent call last):
         ...
        ValueError: reduction too large compared to factor levels

    References
    ----------
    .. [1] Surowiec, Izabella, Ludvig Vikstrom, Gustaf Hector, Erik Johansson,
       Conny Vikstrom, and Johan Trygg. "Generalized Subset Designs in
       Analytical Chemistry." Analytical Chemistry 89, no. 12 (June 20, 2017):
       6491-97. https://doi.org/10.1021/acs.analchem.7b00506.

    .. [2] Vikstrom, Ludvig, Conny Vikstrom, Erik Johansson, and Gustaf Hector.
       Computer-implemented systems and methods for generating
       generalized fractional designs. US9746850 B2, filed May 9,
       2014, and issued August 29, 2017. http://www.google.se/patents/US9746850.

    """
    return _gsd(levels, reduction).generate(n_return_designs = n)
