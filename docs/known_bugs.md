# Current OKL bugs

- There is no check if inner loops are of the same size (one after other). 
    When calculating __launch_bounds__, first loop size is taken into account