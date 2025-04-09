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
            // provide a list of points : a list of list of two floats
            // the list of points are the points of the boundary
            // we guarantee that the initial mesh have those points as nodes
            // only polygonal domain is available so the points are boundary points in 
            // counterclockwise order
        "segments":[[0,1],[1,2],...],
            // mark a list of segments : a list of list of two integers
            // any inner list marks two consecutive points of the boundary
            // homogeneous Dirichlet condition and von Neumann conditions 
            // can only be imposed on marked segments
        "lc":0.1
            // density of the initial mesh
    },
    "dirichlet":[[0,3,4],[1,2]],
        // provide a list of list of integers,
        // any list of integers marks the union of segments,
        // where the Dirichlet BCs are imposed for each load case.
        // for example [[0,3,4],[1,2]] implies that there are two load cases and 
        // first load case has Dirichlet BC over the union of segments 0,3,4
        // the number of Dirichlet BC must be the same as number of VN Bc 
    "vonNeumann":{"loc":[[5,7],[6]],
            // provide a list of list of integers,
            // any list of integers marks the union of segments,
            // where the vonNeumann BCs are imposed.
            // here we have two load cases and the first one
            // has von Neumann located at segment 5 and 7
            "val":[[0.0,0.1],[0.0,0.1]]},
            // list of list of floats, 
            // representing the value of the corresponding vonNeumann bc
            // only constant VN BC is supported by the parser
    
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