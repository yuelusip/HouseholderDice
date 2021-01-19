using QuadGK

"""
    Calculate the asymptotic limit of the cosine similarity
"""
function calc_analytical(zfunc, z_max::Float64)
    function calc_sa_zb(λ,zfunc,a,b)
       v, = quadgk(s-> s^a / (λ-zfunc(s))^b * exp(-s^2/2), -10, 10);
       # display(v)
        return v / sqrt(2*π);
    end

    a, = quadgk(s-> zfunc(s) * s^2 * exp(-s^2/2), -10, 10);
    a /= sqrt(2π);

    θ_list = collect(range(z_max+0.00000000001, z_max+1.5, length=1000));
    s0z1 = calc_sa_zb.(θ_list, zfunc, 0, 1);
    s0z2 = calc_sa_zb.(θ_list, zfunc, 0, 2);
    s2z1 = calc_sa_zb.(θ_list, zfunc, 2, 1);
    s2z2 = calc_sa_zb.(θ_list, zfunc, 2, 2);

    α_ary_dense = s2z1 ./ (s2z1-s0z1);
    μ_list = 1 ./ (θ_list - 1 ./s2z1 .- a);
    L_list = θ_list + (1 ./α_ary_dense .- 1)./s0z1;
    ρ = (1 .+ (1 ./ α_ary_dense .- 1) .* s0z2./(s0z1.^2)) ./ ((1 ./α_ary_dense .- 1) .* s0z2 ./(s0z1.^2) + s2z2 ./(s2z1.^2));

    sel = ρ .< 0;
    ρ[sel] .= 0.0;

    return α_ary_dense, ρ
end
