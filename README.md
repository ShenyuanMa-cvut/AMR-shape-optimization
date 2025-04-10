# An experimental implementation of the Adaptive mesh refinement strategy
## Implementation
The implementation uses `fenicsx` project and `cvxpy`. Solving the following basic algorithm:
1. Define initial mesh and define all the boundary conditions
2. Mesh refinement loop:
    1. inner optimization loop:
        1. solve linear elasticity
        2. use cvxpy to update microstructure by SDP (in parallel and one SDP per triangle)
    2. compute error indicator
    3. refine

## User interface
The user can either construct `ElasticitySolver` by themself or use a simple parser in `demo` to define experiments but there is less freedom in defining the optimization instance. To use the parser in `demo/utils.py` the user should provide a `json` file describing the geometry, boundary conditions and material properties. The `json` file should look like this:

```json
{
    "name":"cantilever_uniform",

    "geometry":{
        "points":[[0.0,0.0],[1.0,0.0],...], 
        "segments":[[0,1],[1,2],...],
        "lc":0.1
    },

    "dirichlet":[[0,3,4],[1,2]],

    "vonNeumann":{
        "loc":[[5,7],[6]],
        "val":[[0.0,0.1],[0.0,0.1]]
    },
    
    "material":{"mu":0.5,"lmbda":0.5},
    
    "opt_param":{
        "l":...,
        "delta":...,
        "inner_itermax":...,
        "ncells_max":...,
        "is_adaptive":true
    }
}
```

- `name` : this field contains the name of the experiments
- `geometry` : this field describes the geometry of the domain, assumed to be a polygonal domain without self intersection.
    - `points` : a list of list of two floats, which are the coordinates of the points thare are present on the boundary of the polygonal domain. The polygonal domain will be drawn by connecting the points in this field in counterclockwise order
    - `segments` : list of list of two integers, this field marks some segments connecting two points of the boundary. If point `i` and `j` are connected then it is assumed that the segment connecting them are part of the boundary.
    - `lc` : float for the density of the cells for initial mesh
- `dirichlet` : list of list of integers. Any inner list of integers marks the union of segments, where the homogeneous Dirichlet BC in prescribed for the corresponding load case. For example `[[0,3,4],[1,2]]` implies that there are two load cases and the first load case has Dirichlet BC over the union of segments 0,3,4. The number of Dirichlet BC must match with the number of VN BC
- `vonNeumann` : this field describes the BC of the load cases.
    - `loc` : list of list of integers and represents the location of the vonNeumann BCs. Similar to `dirichlet`.
    - `val` : list of list of two floats and represents the value of the corresponding VN BC.
- `material` : material property with field `mu` and `lmbda`.
- `opt_param` : parameter of the optimization loops.


