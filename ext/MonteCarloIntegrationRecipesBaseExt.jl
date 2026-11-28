module MonteCarloIntegrationRecipesBaseExt

using MonteCarloIntegration: VEGASResult
using RecipesBase: @recipe, @series

@recipe function f(result::VEGASResult)
    adaptive_grid = result.adaptive_grid
    dimensions = size(adaptive_grid, 2)
    dims = pop!(plotattributes, :dims, 1:min(2, dimensions))
    if !(
            dims isa Union{Tuple, AbstractVector} && 1 <= length(dims) <= 2 &&
                all(dim -> dim isa Integer && !(dim isa Bool) && 1 <= dim <= dimensions, dims) &&
                allunique(dims)
        )
        throw(ArgumentError("dims must select one or two distinct integer dimensions in 1:$dimensions"))
    end

    xdim = first(dims)
    xedges = adaptive_grid[:, xdim]
    xlabel --> "Dimension $xdim"
    xlims --> extrema(xedges)
    grid --> false
    label --> ""

    if length(dims) == 2
        ydim = last(dims)
        yedges = adaptive_grid[:, ydim]
        ylabel --> "Dimension $ydim"
        ylims --> extrema(yedges)
        @series begin
            seriestype := :hline
            yedges
        end
    else
        yshowaxis --> false
        ylims --> (0, 1)
    end

    seriestype := :vline
    xedges
end

end
