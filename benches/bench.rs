use ark_bls12_381::*;
use ark_ff::{One, PrimeField, UniformRand, Zero};
use ark_poly::{univariate::DensePolynomial as DensePoly, univariate::DenseOrSparsePolynomial, UVPolynomial, Polynomial, EvaluationDomain, Evaluations as EvaluationsOnDomain,
    GeneralEvaluationDomain};
//use ark_poly_commit::marlin_pc::MarlinKZG10;
use ark_poly_commit::kzg10;
use ark_poly_commit::kzg10::*;
//use ark_sponge::poseidon::PoseidonSponge;
//use blake2::Blake2s;
//use ark_poly_commit::PolynomialCommitment;
use ark_ec::msm::{FixedBaseMSM, VariableBaseMSM};
use ark_ec::{group::Group, AffineCurve, PairingEngine, ProjectiveCurve};
//use ark_std::rand::rngs::StdRng;
//use ark_std::error::Error;
use ark_std::*;
use rayon::prelude::*;
use ark_relations::r1cs::{
    SynthesisError,
};


const NUM_REPETITIONS: usize = 1;

/*
 * Benchmark parameters
 * t: length of stream's portion for which one generates a proof
 * H: size of the relation to be proven
*/

//const BENCHSIZE: usize = 1 << 11;

fn convert_to_bigints<F: PrimeField>(p: &[F]) -> Vec<F::BigInt> {
    let to_bigint_time = start_timer!(|| "Converting polynomial coeffs to bigints");
    let coeffs = ark_std::cfg_iter!(p)
        .map(|s| s.into_repr())
        .collect::<Vec<_>>();
    end_timer!(to_bigint_time);
    coeffs
}

macro_rules! hsnp_bench {
    ($bench_name:ident, $bench_field:ty, $bench_pairing_engine:ty, $bench_size:ident) => {
        let rng = &mut ark_std::test_rng();
        //type Sponge_Bls12_381 = PoseidonSponge<$bench_pairing_engine>;
        type Poly = DensePoly<$bench_field>;
        
        type PC = kzg10::KZG10<$bench_pairing_engine, Poly>;
        type Pow<'a> = kzg10::Powers<'a, $bench_pairing_engine>;
        type VKt = kzg10::VerifierKey<$bench_pairing_engine>;
        type G1 = G1Projective;

        let t = $bench_size;
        let H  = 4*t;
        //println!("H - t = {}", H - t);
        let max_degree = 2*H;
        let supported_degree = H;

        let pp = PC::setup(max_degree, false, rng).unwrap();
        
        //Trimming the universal params
        let powers_of_g = pp.powers_of_g[..=supported_degree].to_vec();
        let powers_of_gamma_g = (0..=supported_degree)
                .map(|i| pp.powers_of_gamma_g[&i])
                .collect();
        let ck = Pow {
                    powers_of_g: ark_std::borrow::Cow::Owned(powers_of_g),
                    powers_of_gamma_g: ark_std::borrow::Cow::Owned(powers_of_gamma_g),
        };
        let vk = VKt {
                g: pp.powers_of_g[0],
                    gamma_g: pp.powers_of_gamma_g[&0],
                    h: pp.h,
                    beta_h: pp.beta_h,
                    prepared_h: pp.prepared_h.clone(),
                    prepared_beta_h: pp.prepared_beta_h.clone(),
        };
        
        type EvDomain = GeneralEvaluationDomain<$bench_field>;
        let domain_t = EvDomain::new(t)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge).unwrap();
        let domain_h = EvDomain::new(H)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge).unwrap();
        //println!("Size of domain {}", domain_h.size());
        let ZT = Poly::rand(t, rng); //Poly::from_coefficients_vec(vec![-Fr::rand(rng), Fr::one()]);
        
        let ZT_evals = domain_h.coset_fft(&ZT);
        
        /*
        //test polynomial division
        // build c = aZ+b
        let pol_b = Poly::rand(H-1, rng);
        let pol_a = Poly::rand(H-t-1, rng);
        let pol_c = &(&pol_a * &ZT) + &pol_b;
        
        let div_start = ark_std::time::Instant::now();

        let b_evals = domain_h.fft(&pol_b);
        let c_evals = domain_h.fft(&pol_c);

        let mut a_evals = cfg_into_iter!(0..domain_h.size())
            .map(|k| {
                (c_evals[k] - &b_evals[k])/&ZT_evals[k]
            })
            .collect();
        let pol_ap = EvaluationsOnDomain::from_vec_and_domain(a_evals, domain_h)
            .interpolate();
        
        println!(
                "Division test {} in time {} ms",
                (pol_a == pol_ap),
                div_start.elapsed().as_millis() as u128
            );
        
        let divpq_start = ark_std::time::Instant::now();
        let (pol_app, _) = DenseOrSparsePolynomial::from(&pol_c).divide_with_q_and_r(&DenseOrSparsePolynomial::from(&ZT)).unwrap();
        println!(
            "Std Division test {} in time {} ms",
            (pol_a == pol_app),
            divpq_start.elapsed().as_millis() as u128
        );
        */
        
        //set 0 polynomial
        let p0 = Poly::zero(); // Poly::from_coefficients_vec(zero_coeffs);
        
        //sample w' as a random vector
        
        let x = Poly::rand(H-1, rng);
        let wrand = Poly::rand(H-1-t, rng);
        let wprime = &(&wrand * &ZT) + &x;
        
        //Generate simulated signatures
        let mut sigma = Vec::new();
        let mut sigma_r = Vec::new();
            for _ in 1..=t {
                sigma.push(G1::rand(rng).into_affine());
                sigma_r.push(Fr::rand(rng));
            }
        
        let start = ark_std::time::Instant::now();

        for _ in 0..NUM_REPETITIONS {
            
            /* HSNP Prover
             * - c_x <-- Com(x) hiding bound 0
             * - Marlin Prove
             * - CPsvec Prove
             * -- compute w = (w' - x)/Z_T
             * -- Com(w) hiding bound 2
             * -- r0 = r_w' - r_x - r_w * Z_T
             * -- Com(0;r0) ~ Com(0) hiding bound t+2
             * -- c_q <-- EvalProof c0 on random point \rho
             * --- q = (r_0(X) - r_0(\rho))/(X - \rho)
             * --- com_q = Com(0;q)
             * ========================
             * - CPev Prove
             * -- EvalProof c_x
             * -- ...
             * ========================
             * - HSNP-ip Eval on random s
             * - HSNP-ip Eval on 1
             */
            
            
            //=================================
            //c_x <-- Com(x) hiding bound 0
            //=================================

            //sample random polynomial 
            let xsim = Poly::rand(t, rng);
            let (com_x, opn_x) = PC::commit(&ck, &xsim, Some(0), Some(rng)).unwrap();

            //=================================
            //CPsvec Prove
            //=================================

            //compute w = (w' - x)/Z_T using FFT and interpolation on a coset
            // as Z_T is 0 in some elements of the interpolation domain

            
            let div_time = start_timer!(|| "Computing division");

            //evaluate x and wprime on domain's coset of size H
            let mut x_evals = domain_h.coset_fft(&x);
            let mut w_evals = domain_h.coset_fft(&wprime);
            //next compute (w'(h) - x(h))/ZT(h) on elements h of coset
            w_evals = cfg_into_iter!(0..domain_h.size())
            .map(|k| {
                (w_evals[k] - &x_evals[k])/&ZT_evals[k]
            })
            .collect();
            //compute w by interpolation in the coset
            domain_h.coset_ifft_in_place(&mut w_evals);
            let w = Poly::from_coefficients_vec(w_evals);
            //println!("testing w {} -- degree of w {}", (w == wrand), w.degree());

            //let (w, _) = DenseOrSparsePolynomial::from(&w).divide_with_q_and_r(&divisor).unwrap();
//            let w = &(&wprime - &x) / &ZT; // this is very expensive; see how to optimize
            // Idea: note that for the true witness w' - x = \sum_{h \in H \ T} w'(h)L_h(X)/Z_T(x)
            //  So one could precompute the group elements [L_h(s)/Z_T(s)] for all h \in H \ T
            end_timer!(div_time);
            
            //-- Com(w) hiding bound 2
            let (com_w, opn_w) = PC::commit(&ck, &w, Some(2), Some(rng)).unwrap();

            //-- Com(0;r0) ~ Com(0) hiding bound t+2
            let (com_0, opn_0) = PC::commit(&ck, &p0, Some(t+2), Some(rng)).unwrap();
            
            let rho = Fr::rand(rng);
            //-- c_q <-- EvalProof c0 on random point \rho
            //let value = p0.evaluate(&rho);
            let cq0 = PC::open(&ck, &p0, rho, &opn_0).unwrap();

            //=================================
            //CPev Prove
            //=================================

            let r = Fr::rand(rng);
            let xr = PC::open(&ck, &x, r, &opn_x).unwrap();

            //add the additional bits

            //=================================
            //- HSNP-ip Eval on random s
            //=================================

            let mut s = Vec::new();
            for _ in 1..=t {
                s.push(Fr::rand(rng));
            }

            let s_ints = convert_to_bigints(&s);
            let sigmaev = VariableBaseMSM::multi_scalar_mul(&sigma, s_ints.as_slice());
            let mut sigma_r_ev = sigma_r[0]*s[0];
            for i in 1..=t-1 {
                sigma_r_ev += sigma_r[i]*s[i];
            }


            //=================================
            // - HSNP-ip Eval on 1
            //=================================

            let mut sigmaev1 = sigma[0].into_projective();
            let mut sigma_r_ev1 = sigma_r[0];
            for i in 1..=t-1 {
                sigmaev1 += sigma[i].into_projective();
                sigma_r_ev1 += sigma_r[i];
            }

        }
        

        println!(
            "proving time for {} and t={} and H={} : {} ms",
            stringify!($bench_pairing_engine),
            $bench_size,
            ($bench_size*4),
            start.elapsed().as_millis() / NUM_REPETITIONS as u128
        );
    };
}

fn bench_prove() {
    let benchsize: usize = 1 << 10;
    hsnp_bench!(bls, Fr, Bls12_381, benchsize);
}

fn main() {
    println!(
        "Running HSNP benchmarks for inputs of size t=..."
    );
    bench_prove();
}

