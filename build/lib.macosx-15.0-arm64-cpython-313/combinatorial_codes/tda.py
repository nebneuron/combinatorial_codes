import gudhi
import numpy as np

def compute_homology_from_facets(facets, max_dimension=np.inf):
    """
    Compute the persistence diagram and count the number of infinite bars by dimension.
    persistence, infinite_bar_counts = compute_homology_from_facets(facets, max_dimension=2)
    """

    st = gudhi.SimplexTree() # initializes a simplex tree

    # Insert each facet; Gudhi automatically adds lower-dimensional faces.
    for facet in facets:
        st.insert(facet)

    # Ensure the filtration is non-decreasing. 
    st.make_filtration_non_decreasing()
    if not np.isinf(max_dimension):
        st.prune_above_dimension(max_dimension) # remove simplices above given dimension, if we don't want to compute all possible homology groups

    persistence = st.persistence(persistence_dim_max=True) # persistence_dim_max=True tells to compute all homology groups up to maximal dimension

    # Extract infinite bars (features with death == infinity).
    infinite_bars = [(dim, (birth, death))
                     for dim, (birth, death) in persistence if death == float("inf")]
    
     # Count infinite bars by dimension.
    infinite_bar_counts = {}
    for dim, _ in infinite_bars:
        infinite_bar_counts[dim] = infinite_bar_counts.get(dim, 0) + 1
    
    return persistence, infinite_bar_counts


def homology_is_trivial(facets, max_dimension=np.inf): 
    """
    Check if the homology of the simplicial complex defined by the facets is that of a contractible space.
    homology_is_trivial(facets)

    Example:        
    facets = [[0, 1], [1, 2],[2, 3], [0, 3], [0,5], [5,6],[6,0], [10,11]]
    homology_is_trivial(facets) # False (the space is not contractible)
    """
    if len(facets) == 0:
        raise ValueError("Facets must be non-empty.")
    
    st = gudhi.SimplexTree() # initializes a simplex tree

    # Insert each facet; Gudhi automatically adds lower-dimensional faces.
    for facet in facets:
        st.insert(facet)

    # Ensure the filtration is non-decreasing. 
    st.make_filtration_non_decreasing()
    if not np.isinf(max_dimension):
        st.prune_above_dimension(max_dimension) # remove simplices above given dimension, if we don't want to compute all possible homology groups

    st.persistence(persistence_dim_max=True) # persistence_dim_max=True tells to compute all homology groups up to maximal dimension
    Betti_numbers = st.betti_numbers()
    return (Betti_numbers[0] == 1) and (sum(Betti_numbers[1:])==0)

