//! Here we construct a polynomial commitment that enables users to commit to a
//! single polynomial `p`, and then later provide an evaluation proof that
//! convinces verifiers that a claimed value `v` is the true evaluation of `p`
//! at a chosen point `x`. Our construction follows the template of the construction
//! proposed by Kate, Zaverucha, and Goldberg ([KZG11](http://cacr.uwaterloo.ca/techreports/2010/cacr2010-10.pdf)).
//! This construction achieves extractability in the algebraic group model (AGM).

use crate::{BTreeMap, Error, PCRandomness, Vec};
use ark_ec::msm::{FixedBaseMSM, VariableBaseMSM};
use ark_ec::{group::Group, AffineCurve, PairingEngine, ProjectiveCurve};
use ark_ff::{One, PrimeField, UniformRand, Zero, FftField, FromBytes};
use ark_poly::{UVPolynomial, univariate::DensePolynomial as DensePoly, GeneralEvaluationDomain, EvaluationDomain};
use ark_std::{format, marker::PhantomData, ops::Div, vec};

use ark_std::rand::{RngCore, SeedableRng};

use rand_chacha::ChaChaRng;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use ark_ff::to_bytes;
//use ark_marlin::rng::FiatShamirRng;
//use digest::Digest;
use blake2::{Blake2s, Digest};

mod data_structures;
pub use data_structures::*;

/// `HSNP_PC` is an implementation of the polynomial commitment scheme of
/// [Kate, Zaverucha and Goldbgerg][kzg10]
///
/// [kzg10]: http://cacr.uwaterloo.ca/techreports/2010/cacr2010-10.pdf
pub struct HsnpPc<E: PairingEngine, P: UVPolynomial<E::Fr>> {
    _engine: PhantomData<E>,
    _poly: PhantomData<P>,
}

impl<E, P> HsnpPc<E, P>
where
    E: PairingEngine,
    P: UVPolynomial<E::Fr, Point = E::Fr>,
    for<'a, 'b> &'a P: Div<&'b P, Output = P>,
{
    /// protocol name for CPev, to be used to seed FiatShamir rng
    pub const CPEV_NAME: &'static [u8] = b"HSNP_PC-CPev";
    /// protocol name for PoK protocol, to be used to seed FiatShamir rng
    pub const POK_NAME: &'static [u8] = b"HSNP_PC-PoK";

    /// Constructs public parameters when given as input the maximum degree `degree`
    /// for the polynomial commitment scheme.
    pub fn setup<R: RngCore>(
        max_degree: usize,
        produce_g2_powers: bool,
        rng: &mut R,
    ) -> Result<UniversalParams<E>, Error> {
        if max_degree < 1 {
            return Err(Error::DegreeIsZero);
        }
        let setup_time = start_timer!(|| format!("HSNP_PC::Setup with degree {}", max_degree));
        let beta = E::Fr::rand(rng);
        let g = E::G1Projective::rand(rng);
        let gamma_g = E::G1Projective::rand(rng);
        let h = E::G2Projective::rand(rng);

        let mut powers_of_beta = vec![E::Fr::one()];

        let mut cur = beta;
        for _ in 0..max_degree {
            powers_of_beta.push(cur);
            cur *= &beta;
        }

        let window_size = FixedBaseMSM::get_mul_window_size(max_degree + 1);

        let scalar_bits = E::Fr::size_in_bits();
        let g_time = start_timer!(|| "Generating powers of G");
        let g_table = FixedBaseMSM::get_window_table(scalar_bits, window_size, g);
        let powers_of_g = FixedBaseMSM::multi_scalar_mul::<E::G1Projective>(
            scalar_bits,
            window_size,
            &g_table,
            &powers_of_beta,
        );
        end_timer!(g_time);
        let gamma_g_time = start_timer!(|| "Generating powers of gamma * G");
        let gamma_g_table = FixedBaseMSM::get_window_table(scalar_bits, window_size, gamma_g);
        let powers_of_gamma_g = FixedBaseMSM::multi_scalar_mul::<E::G1Projective>(
            scalar_bits,
            window_size,
            &gamma_g_table,
            &powers_of_beta,
        );
        // Add an additional power of gamma_g, because we want to be able to support
        // up to D queries. -- REMOVED
        //powers_of_gamma_g.push(powers_of_gamma_g.last().unwrap().mul(&beta));
        end_timer!(gamma_g_time);

        let powers_of_g = E::G1Projective::batch_normalization_into_affine(&powers_of_g);
        let powers_of_gamma_g =
            E::G1Projective::batch_normalization_into_affine(&powers_of_gamma_g)
                .into_iter()
                .enumerate()
                .collect();

        let powers_of_h_time = start_timer!(|| "Generating negative powers of h in G2");
        let powers_of_h = if produce_g2_powers {
            let mut powers_of_beta = vec![E::Fr::one()];
            let mut cur = beta;
            for _ in 0..max_degree {
                powers_of_beta.push(cur);
                cur *= &beta;
            }

            let h_table = FixedBaseMSM::get_window_table(scalar_bits, window_size, h);
            let powers_of_h = FixedBaseMSM::multi_scalar_mul::<E::G2Projective>(
                scalar_bits,
                window_size,
                &h_table,
                &powers_of_beta,
            );

            let affines = E::G2Projective::batch_normalization_into_affine(&powers_of_h);
            let mut affines_map = BTreeMap::new();
            affines.into_iter().enumerate().for_each(|(i, a)| {
                affines_map.insert(i, a);
            });
            affines_map
        } else {
            BTreeMap::new()
        };

        end_timer!(powers_of_h_time);

        let h = h.into_affine();
        let beta_h = h.mul(beta).into_affine();
        let prepared_h = h.into();
        let prepared_beta_h = beta_h.into();

        let pp = UniversalParams {
            powers_of_g,
            powers_of_gamma_g,
            h,
            beta_h,
            powers_of_h,
            prepared_h,
            prepared_beta_h,
        };
        end_timer!(setup_time);
        Ok(pp)
    }

    /// takes a subset of the universal parameters to let the
    /// commitment work with polynomials up to a given degree
    /// and hiding bound
    pub fn trim(
        pp: &UniversalParams<E>,
        mut supported_degree: usize,
        max_hiding_bound: usize
    ) -> Result<(Powers<E>, VerifierKey<E>), Error> {
        if supported_degree == 1 {
            supported_degree += 1;
        }
        let powers_of_g = pp.powers_of_g[..=supported_degree].to_vec();
        let powers_of_gamma_g = (0..=max_hiding_bound + 1)
            .map(|i| pp.powers_of_gamma_g[&i])
            .collect();

        let powers = Powers {
            powers_of_g: ark_std::borrow::Cow::Owned(powers_of_g),
            powers_of_gamma_g: ark_std::borrow::Cow::Owned(powers_of_gamma_g),
        };
        let vk = VerifierKey {
            g: pp.powers_of_g[0],
            gamma_g: pp.powers_of_gamma_g[&0],
            gamma_beta_g: pp.powers_of_gamma_g[&1],
            h: pp.h,
            beta_h: pp.beta_h,
            prepared_h: pp.prepared_h.clone(),
            prepared_beta_h: pp.prepared_beta_h.clone(),
        };
        Ok((powers, vk))
    }

    
    /// Outputs specialized proving key and verification key for the CPsvec CPSNARK
    /// The specialization is with respect to the number t of signed inputs;
    /// the proving key consists of the vanishing polynomial in domain_t, which
    /// are the first t elements of the multiplciative subgroup H of size n;
    /// the specialized verification key consists of a deterministic commitment in
    /// G2 of this vanishing polynomial, i.e., g_2^{z_t(s)}.
    pub fn specialize(
        pp: &UniversalParams<E>,
        vk: &VerifierKey<E>,
        n: usize,
        t: usize,
    ) -> Result<(DensePoly<E::Fr>, SpecializedVerifierKey<E>), Error> {
        
         //generate FFT interpolation domain of size |H|=num_constraints
        //(this is the same interpolation domain of Marlin)
        let domain_h = GeneralEvaluationDomain::<E::Fr>::new(n).unwrap();
        //println!("Size of domain {}", domain_h.size());
        
        //Define the domain T that contains the first t elements of H
        let mut domain_t = Vec::new();
        for k in 0..t {
            domain_t.push(domain_h.element(k));
        }
        let z_t = compute_vanishing_polynomial(&domain_t);
        
        let (_, plain_coeffs) =
            skip_leading_zeros_and_convert_to_bigints(&z_t);

        let mut powers_of_h = Vec::new();
        for i in 0..=t {
            powers_of_h.push(pp.powers_of_h[&i]);
        }
        let c_z = VariableBaseMSM::multi_scalar_mul(
            &powers_of_h,
            &plain_coeffs,
        ); //need to negate this
        
        let vk_t = SpecializedVerifierKey {
            vk: vk.clone(),
            com_zt: c_z.into_affine(),
        };
        Ok((z_t, vk_t))
    }

    ///Outputs the elements of the public parameters that are necessary to
    /// compute a commitment to a scalar, as well as the generator of G2
    pub fn get_scalar_comkey(
        pp: &UniversalParams<E>,
    ) -> Result<ScalarComKey<E>, Error> {

        let ck = ScalarComKey {
            g: pp.powers_of_g[0],
            gamma_g: pp.powers_of_gamma_g[&0],
            h: pp.h,
        };
        Ok(ck)
    }

    /// Compute a vector of commitments to all the lagrange polynomials in a domain
    /// (taken almost the same from https://github.com/zcash/mpc/blob/master/src/protocol/qap.rs)
    pub fn compute_lagrange_comkey<G: Group + Group<ScalarField = F>, F: FftField + PrimeField>(
        vec: &[G], omega: F, size: usize
    ) -> Vec<G> {
        
        assert!(vec.len() >= 2);
        assert_eq!((vec.len() / 2) * 2, vec.len());
        
        //let mut tmp = Vec::<G>::new();
        
        let domain_size_inverse = F::from(size as u64).inverse().unwrap();
        let mut tmp = fft(vec, omega);
        tmp.reverse(); // coefficients are in reverse

        //mul_all_by(&mut tmp, domain_size_inverse);
        for k in 0..tmp.len() {
            tmp[k] = tmp[k].mul(&domain_size_inverse);
        }
        
        tmp
    }

    /// Outputs a commitment to `polynomial`.
    pub fn commit(
        powers: &Powers<E>,
        polynomial: &P,
        hiding_bound: Option<usize>,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<(Commitment<E>, Randomness<E::Fr, P>), Error> {
        Self::check_degree_is_too_large(polynomial.degree(), powers.size())?;

        let commit_time = start_timer!(|| format!(
            "Committing to polynomial of degree {} with hiding_bound: {:?}",
            polynomial.degree(),
            hiding_bound,
        ));

        let (num_leading_zeros, plain_coeffs) =
            skip_leading_zeros_and_convert_to_bigints(polynomial);

        let msm_time = start_timer!(|| "MSM to compute commitment to plaintext poly");
        let mut commitment = VariableBaseMSM::multi_scalar_mul(
            &powers.powers_of_g[num_leading_zeros..],
            &plain_coeffs,
        );
        end_timer!(msm_time);

        let mut randomness = Randomness::<E::Fr, P>::empty();
        if let Some(hiding_degree) = hiding_bound {
            let mut rng = rng.ok_or(Error::MissingRng)?;
            let sample_random_poly_time = start_timer!(|| format!(
                "Sampling a random polynomial of degree {}",
                hiding_degree
            ));

            randomness = Randomness::rand(hiding_degree, false, None, &mut rng);
            Self::check_hiding_bound(
                randomness.blinding_polynomial.degree(),
                powers.powers_of_gamma_g.len(),
            )?;
            end_timer!(sample_random_poly_time);
        }

        let random_ints = convert_to_bigints(&randomness.blinding_polynomial.coeffs());
        let msm_time = start_timer!(|| "MSM to compute commitment to random poly");
        let random_commitment =
            VariableBaseMSM::multi_scalar_mul(&powers.powers_of_gamma_g, random_ints.as_slice())
                .into_affine();
        end_timer!(msm_time);

        commitment.add_assign_mixed(&random_commitment);

        end_timer!(commit_time);
        Ok((Commitment(commitment.into()), randomness))
    }

    ///Generate a commitment with a given randomness
    pub fn deterministic_commit(
        powers: &Powers<E>,
        polynomial: &P,
        rand: &P,
    ) -> Result<(Commitment<E>, Randomness<E::Fr, P>), Error> {
        Self::check_degree_is_too_large(polynomial.degree(), powers.size())?;

        let commit_time = start_timer!(|| format!(
            "Committing to polynomial of degree {} with randomness of degree {}",
            polynomial.degree(),
            rand.degree(),
        ));

        let (num_leading_zeros, plain_coeffs) =
            skip_leading_zeros_and_convert_to_bigints(polynomial);

        let msm_time = start_timer!(|| "MSM to compute commitment to plaintext poly");
        let mut commitment = VariableBaseMSM::multi_scalar_mul(
            &powers.powers_of_g[num_leading_zeros..],
            &plain_coeffs,
        );
        end_timer!(msm_time);

        //change from here
        let mut randomness = Randomness::<E::Fr, P>::empty();
        randomness.blinding_polynomial = rand.clone();
        
        let random_ints = convert_to_bigints(&randomness.blinding_polynomial.coeffs());
        let msm_time = start_timer!(|| "MSM to compute commitment to random poly");
        let random_commitment =
            VariableBaseMSM::multi_scalar_mul(&powers.powers_of_gamma_g, random_ints.as_slice())
                .into_affine();
        end_timer!(msm_time);

        commitment.add_assign_mixed(&random_commitment);

        end_timer!(commit_time);

        Ok((Commitment(commitment.into()), randomness))
    }

    /// Compute commitment to a polynomial taken as input as vector of coefficients in Lagrange basis
    pub fn commit_from_lagrange_representation(
        powers: &Powers<E>,
        ck_t: &[E::G1Affine],
        vec: &[E::Fr],
        hiding_bound: Option<usize>,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<(Commitment<E>, Randomness<E::Fr, P>), Error> {
        Self::check_degree_is_too_large(vec.len(), powers.size())?;

        let commit_time = start_timer!(|| format!(
            "Committing to vector of size {} with hiding_bound: {:?}",
            vec.len(),
            hiding_bound,
        ));

        let int_coeffs = convert_to_bigints(vec);
        let mut commitment = VariableBaseMSM::multi_scalar_mul(
            &ck_t,
            &int_coeffs,
        );

        let mut randomness = Randomness::<E::Fr, P>::empty();
        if let Some(hiding_degree) = hiding_bound {
            let mut rng = rng.ok_or(Error::MissingRng)?;
            let sample_random_poly_time = start_timer!(|| format!(
                "Sampling a random polynomial of degree {}",
                hiding_degree
            ));

            randomness = Randomness::rand(hiding_degree, false, None, &mut rng);
            Self::check_hiding_bound(
                randomness.blinding_polynomial.degree(),
                powers.powers_of_gamma_g.len(),
            )?;
            end_timer!(sample_random_poly_time);
        }

        let random_ints = convert_to_bigints(&randomness.blinding_polynomial.coeffs());
        let msm_time = start_timer!(|| "MSM to compute commitment to random poly");
        let random_commitment =
            VariableBaseMSM::multi_scalar_mul(&powers.powers_of_gamma_g, random_ints.as_slice())
                .into_affine();
        end_timer!(msm_time);

        commitment.add_assign_mixed(&random_commitment);

        end_timer!(commit_time);
        Ok((Commitment(commitment.into()), randomness))
    }

    /// Compute witness polynomial.
    ///
    /// The witness polynomial w(x) the quotient of the division (p(x) - p(z)) / (x - z)
    /// Observe that this quotient does not change with z because
    /// p(z) is the remainder term. We can therefore omit p(z) when computing the quotient.
    pub fn compute_witness_polynomial(
        p: &P,
        point: P::Point,
        randomness: &Randomness<E::Fr, P>,
    ) -> Result<(P, Option<P>), Error> {
        let divisor = P::from_coefficients_vec(vec![-point, E::Fr::one()]);

        let witness_time = start_timer!(|| "Computing witness polynomial");
        let witness_polynomial = p / &divisor;
        end_timer!(witness_time);

        let random_witness_polynomial = if randomness.is_hiding() {
            let random_p = &randomness.blinding_polynomial;

            let witness_time = start_timer!(|| "Computing random witness polynomial");
            let random_witness_polynomial = random_p / &divisor;
            end_timer!(witness_time);
            Some(random_witness_polynomial)
        } else {
            None
        };

        Ok((witness_polynomial, random_witness_polynomial))
    }

    pub(crate) fn open_with_witness_polynomial<'a>(
        powers: &Powers<E>,
        point: P::Point,
        randomness: &Randomness<E::Fr, P>,
        witness_polynomial: &P,
        hiding_witness_polynomial: Option<&P>,
    ) -> Result<Proof<E>, Error> {
        Self::check_degree_is_too_large(witness_polynomial.degree(), powers.size())?;
        let (num_leading_zeros, witness_coeffs) =
            skip_leading_zeros_and_convert_to_bigints(witness_polynomial);

        let witness_comm_time = start_timer!(|| "Computing commitment to witness polynomial");
        let mut w = VariableBaseMSM::multi_scalar_mul(
            &powers.powers_of_g[num_leading_zeros..],
            &witness_coeffs,
        );
        end_timer!(witness_comm_time);

        let random_v = if let Some(hiding_witness_polynomial) = hiding_witness_polynomial {
            let blinding_p = &randomness.blinding_polynomial;
            let blinding_eval_time = start_timer!(|| "Evaluating random polynomial");
            let blinding_evaluation = blinding_p.evaluate(&point);
            end_timer!(blinding_eval_time);

            let random_witness_coeffs = convert_to_bigints(&hiding_witness_polynomial.coeffs());
            let witness_comm_time =
                start_timer!(|| "Computing commitment to random witness polynomial");
            w += &VariableBaseMSM::multi_scalar_mul(
                &powers.powers_of_gamma_g,
                &random_witness_coeffs,
            );
            end_timer!(witness_comm_time);
            Some(blinding_evaluation)
        } else {
            None
        };

        Ok(Proof {
            w: w.into_affine(),
            random_v,
        })
    }

    /// On input a polynomial `p` and a point `point`, outputs a proof for the same.
    pub fn open<'a>(
        powers: &Powers<E>,
        p: &P,
        point: P::Point,
        rand: &Randomness<E::Fr, P>,
    ) -> Result<Proof<E>, Error> {
        Self::check_degree_is_too_large(p.degree(), powers.size())?;
        let open_time = start_timer!(|| format!("Opening polynomial of degree {}", p.degree()));

        let witness_time = start_timer!(|| "Computing witness polynomials");
        let (witness_poly, hiding_witness_poly) = Self::compute_witness_polynomial(p, point, rand)?;
        end_timer!(witness_time);

        let proof = Self::open_with_witness_polynomial(
            powers,
            point,
            rand,
            &witness_poly,
            hiding_witness_poly.as_ref(),
        );

        end_timer!(open_time);
        proof
    }

    /// On input a polynomial `p` and an opening `rand_p`,
    /// along with an opening `rand_z` of a commitment to z=p(point),
    /// and an evaluation point `point`, outputs a proof for p(point)=z.
    pub fn committed_evaluation_prove<R: RngCore>(
        powers: &Powers<E>,
        vk: &VerifierKey<E>,
        point: P::Point,
        com_p: &Commitment<E>,
        p: &P,
        rand_p: &Randomness<E::Fr, P>,
        com_z: &Commitment<E>,
        z: &P,
        rand_z: &Randomness<E::Fr, P>,
        rng: &mut R,
    ) -> Result<ComEvProof<E>, Error> {
        Self::check_degree_is_too_large(p.degree(), powers.size())?;
        
        let witness_time = start_timer!(|| "Computing witness polynomials");
        let divisor = P::from_coefficients_vec(vec![-point, E::Fr::one()]);
        let q = p / &divisor;
        end_timer!(witness_time);

        let (num_leading_zeros, witness_coeffs) = skip_leading_zeros_and_convert_to_bigints(&q);

        let witness_comm_time = start_timer!(|| "Computing commitment to witness polynomial");
        let mut com_q = VariableBaseMSM::multi_scalar_mul(
            &powers.powers_of_g[num_leading_zeros..],
            &witness_coeffs,
        );

        let rand_q = E::Fr::rand(rng);
        let com_r = powers.powers_of_gamma_g[0].mul(rand_q).into_affine();
        com_q.add_assign_mixed(&com_r);
        end_timer!(witness_comm_time);

        //Compute ZKPoK for com_z
        let ck = ScalarComKey {
            g: vk.g,
            gamma_g: vk.gamma_g,
            h: vk.h,
        };
        let pi_z = Self::pok_scalar_prove(&ck, com_z, z, rand_z, rng).unwrap();
        //Compute ZKPoK for e()
        
        //tilde{g} = h_1^{s - y}
        let tilde_g = powers.powers_of_gamma_g[1].into_projective() - &(powers.powers_of_gamma_g[0].mul(point));
        
        
        //Sigma protocol first message U = e(h1^gamma \tilde{g}^delta, g_2)
        let (gamma, delta) = (E::Fr::rand(rng), E::Fr::rand(rng));
        let inner = powers.powers_of_gamma_g[0].mul(gamma) + &(tilde_g.into_affine().mul(delta));
        let u = E::pairing(inner, vk.h);

        //test: compute U_ver = e(h1^gamma, g_2)e(h1^delta, g_2^{s-y})
        //let inner2 = vk.beta_h.into_projective() - &vk.h.mul(point);
        /*
        let in1 = powers.powers_of_gamma_g[0].mul(gamma).into_affine();
        let u_ver = E::product_of_pairings(&[
            (in1.into(), vk.prepared_h.clone()),
            (powers.powers_of_gamma_g[0].mul(delta).into_affine().into(), inner.into_affine().into() ), 
        ]);
        //E::pairing(powers.powers_of_gamma_g[1].mul(gamma), vk.h) * E::pairing(powers.powers_of_gamma_g[1].mul(delta), (vk.beta_h.into_projective() + &vk.h.mul(-point)));
        eprintln!("prover u={}\n\n u_ver={}\n\n\n", u, u_ver);
        */
        
        //compute challenge rho = Hash(identifier, y, com_p, com_z, pi_z, com_q, u)
        //let rho = fiat_shamir_hash::<E,P>(&Self::CPEV_NAME, point, &com_p.0, &com_z.0, &com_q.into_affine(), u);
        let bytes = to_bytes![&Self::CPEV_NAME, point, &com_p.0, &com_z.0, &pi_z.com1, &pi_z.sigma, &pi_z.tau, &com_q.into_affine(), u].unwrap();
        let rho = fiat_shamir_hashfn::<E,P>(&bytes);
        
        //compute sigma protocol's response (sigma, tau)
        //sigma = gamma - rho (opn_z - opn_p)
        let sigma = gamma - rho*(rand_z.blinding_polynomial.coeffs()[0] - rand_p.blinding_polynomial.coeffs()[0]);
        //tau = delta - rho opn_q
        let tau = delta - rho*rand_q;
        
        /*
        //tests
        let test_left = (vk.g.mul(z.coeffs()[0]) + &vk.gamma_g.mul(rand_z.blinding_polynomial.coeffs()[0])).into_affine();
        let test_right = com_z.0;


        //testing here
        let u_chk = E::product_of_pairings(&[
            ((vk.gamma_g.mul(sigma) + tilde_g.into_affine().mul(tau)).into_affine().into(), vk.prepared_h.clone()),
            (com_q.into_affine().mul(rho).into_affine().into(), inner2.into_affine().into()),
            ((com_z.0.into_projective() - &com_p.0.into_projective()).into_affine().into(), vk.h.mul(rho).into_affine().into()),
        ]);
        eprintln!("prover u={}\n\n u_chk={}\n\n\nTest={},rand_z={}", u, u_chk, test_left==test_right, rand_z.blinding_polynomial.coeffs()[0]);
        */
        
        Ok(ComEvProof {
            pi_z,
            com_q: com_q.into_affine(),
            rho,
            sigma,
            tau,
        })
        
    }

    ///Verifies a committed evaluation proof
    pub fn committed_evaluation_ver(
        vk: &VerifierKey<E>,
        point: E::Fr,
        com_p: &Commitment<E>,
        com_z: &Commitment<E>,
        proof: &ComEvProof<E>,
    ) -> Result<bool, Error> {
        
        //tilde{g} = h_1^{s - y}
        let tilde_g = vk.gamma_beta_g.into_projective() + &vk.gamma_g.mul(-point);
        
        // U = e(h1^sigma tilde{g}^delta) A^rho s.t. A = e(c_q, g_2^{s - y}) e(c_p/c_z, g_2)
        
        //inner = h_1^s / h_1^y
        let inner = vk.beta_h.into_projective() - &vk.h.mul(point);

        let u = E::product_of_pairings(&[
            ((vk.gamma_g.mul(proof.sigma) + tilde_g.into_affine().mul(proof.tau)).into_affine().into(), vk.prepared_h.clone()),
            (proof.com_q.mul(proof.rho).into_affine().into(), inner.into_affine().into()),
            ((com_z.0.into_projective() - &com_p.0.into_projective()).into_affine().into(), vk.h.mul(proof.rho).into_affine().into()),
        ]);

        //eprintln!("verifier u={}", u);
        //check that rho = Hash(identifier, y, com_p, com_z, pi_z, com_q, u) for the computed u
        //let rhs = fiat_shamir_hash::<E,P>(&Self::CPEV_NAME, point, &com_p.0, &com_z.0, &proof.com_q, u);

        let bytes = to_bytes![&Self::CPEV_NAME, point, &com_p.0, &com_z.0, &proof.pi_z.com1, &proof.pi_z.sigma, &proof.pi_z.tau, &proof.com_q, u].unwrap();
        let rhs = fiat_shamir_hashfn::<E,P>(&bytes);
        let ck = ScalarComKey {
            g: vk.g,
            gamma_g: vk.gamma_g,
            h: vk.h,
        };
        let ver_pi_z = Self::pok_scalar_ver(&ck, com_z, &proof.pi_z).unwrap();

        Ok(proof.rho == rhs && ver_pi_z)
    }

    ///Generates a ZKPoK of the opening of a commitment to a scalar with hiding bound 0
    pub fn pok_scalar_prove<R: RngCore>(
        //vk: &VerifierKey<E>,
        ck: &ScalarComKey<E>,
        com: &Commitment<E>,
        p: &P,
        opn_p: &Randomness<E::Fr, P>,
        rng: &mut R,
    ) -> Result<PoKProof<E>, Error> {

        Self::check_degree_is_zero(p.degree())?;
        Self::check_degree_is_zero(opn_p.blinding_polynomial.degree())?;
        
        let (gamma, delta) = (E::Fr::rand(rng), E::Fr::rand(rng));
        let first = (ck.g.mul(gamma) + &ck.gamma_g.mul(delta)).into_affine();
        //generate random oracle challenge
        let bytes = to_bytes![&Self::POK_NAME, &com.0, &first].unwrap();
        let rho = fiat_shamir_hashfn::<E,P>(&bytes);
        //sigma protocol response
        let sigma = gamma + rho*p.coeffs()[0];
        let tau = delta + rho*opn_p.blinding_polynomial.coeffs()[0];

        Ok(PoKProof {
            com1: first,
            sigma,
            tau,
        })
    }

    ///Verifies a ZKPoK of the opening of a commitment to a scalar with hiding bound 0
    pub fn pok_scalar_ver(
        //vk: &VerifierKey<E>,
        ck: &ScalarComKey<E>,
        com: &Commitment<E>,
        pi: &PoKProof<E>,
    ) -> Result<bool, Error> {

        //generate random oracle challenge
        let bytes = to_bytes![&Self::POK_NAME, &com.0, &pi.com1].unwrap();
        let rho = fiat_shamir_hashfn::<E,P>(&bytes);
        //check that com1 com^rho = g_1^sigma h_1^tu
        let lhs = ck.g.mul(pi.sigma) + &ck.gamma_g.mul(pi.tau);
        let rhs = pi.com1.into_projective() + &com.0.mul(rho);

        Ok(lhs == rhs)
        
    }

    /// Verifies that `value` is the evaluation at `point` of the polynomial
    /// committed inside `comm`.
    pub fn check(
        vk: &VerifierKey<E>,
        comm: &Commitment<E>,
        point: E::Fr,
        value: E::Fr,
        proof: &Proof<E>,
    ) -> Result<bool, Error> {
        let check_time = start_timer!(|| "Checking evaluation");
        let mut inner = comm.0.into_projective() - &vk.g.mul(value);
        if let Some(random_v) = proof.random_v {
            inner -= &vk.gamma_g.mul(random_v);
        }
        let lhs = E::pairing(inner, vk.h);

        let inner = vk.beta_h.into_projective() - &vk.h.mul(point);
        let rhs = E::pairing(proof.w, inner);

        end_timer!(check_time, || format!("Result: {}", lhs == rhs));
        Ok(lhs == rhs)
    }

    /// Check that each `proof_i` in `proofs` is a valid proof of evaluation for
    /// `commitment_i` at `point_i`.
    pub fn batch_check<R: RngCore>(
        vk: &VerifierKey<E>,
        commitments: &[Commitment<E>],
        points: &[E::Fr],
        values: &[E::Fr],
        proofs: &[Proof<E>],
        rng: &mut R,
    ) -> Result<bool, Error> {
        let check_time =
            start_timer!(|| format!("Checking {} evaluation proofs", commitments.len()));

        let mut total_c = <E::G1Projective>::zero();
        let mut total_w = <E::G1Projective>::zero();

        let combination_time = start_timer!(|| "Combining commitments and proofs");
        let mut randomizer = E::Fr::one();
        // Instead of multiplying g and gamma_g in each turn, we simply accumulate
        // their coefficients and perform a final multiplication at the end.
        let mut g_multiplier = E::Fr::zero();
        let mut gamma_g_multiplier = E::Fr::zero();
        for (((c, z), v), proof) in commitments.iter().zip(points).zip(values).zip(proofs) {
            let w = proof.w;
            let mut temp = w.mul(*z);
            temp.add_assign_mixed(&c.0);
            let c = temp;
            g_multiplier += &(randomizer * v);
            if let Some(random_v) = proof.random_v {
                gamma_g_multiplier += &(randomizer * &random_v);
            }
            total_c += &c.mul(randomizer.into_repr());
            total_w += &w.mul(randomizer.into_repr());
            // We don't need to sample randomizers from the full field,
            // only from 128-bit strings.
            randomizer = u128::rand(rng).into();
        }
        total_c -= &vk.g.mul(g_multiplier);
        total_c -= &vk.gamma_g.mul(gamma_g_multiplier);
        end_timer!(combination_time);

        let to_affine_time = start_timer!(|| "Converting results to affine for pairing");
        let affine_points = E::G1Projective::batch_normalization_into_affine(&[-total_w, total_c]);
        let (total_w, total_c) = (affine_points[0], affine_points[1]);
        end_timer!(to_affine_time);

        let pairing_time = start_timer!(|| "Performing product of pairings");
        let result = E::product_of_pairings(&[
            (total_w.into(), vk.prepared_beta_h.clone()),
            (total_c.into(), vk.prepared_h.clone()),
        ])
        .is_one();
        end_timer!(pairing_time);
        end_timer!(check_time, || format!("Result: {}", result));
        Ok(result)
    }

    pub(crate) fn check_degree_is_too_large(degree: usize, num_powers: usize) -> Result<(), Error> {
        let num_coefficients = degree + 1;
        if num_coefficients > num_powers {
            Err(Error::TooManyCoefficients {
                num_coefficients,
                num_powers,
            })
        } else {
            Ok(())
        }
    }

    pub(crate) fn check_degree_is_zero(degree: usize) -> Result<(), Error> {
        if degree > 0 {
            Err(Error::DegreeIsNotZero)
        } else {
            Ok(())
        }
    }

    pub(crate) fn check_hiding_bound(
        hiding_poly_degree: usize,
        num_powers: usize,
    ) -> Result<(), Error> {
        if hiding_poly_degree >= num_powers {
            // The above check uses `>=` because committing to a hiding poly with
            // degree `hiding_poly_degree` requires `hiding_poly_degree + 1`
            // powers.
            Err(Error::HidingBoundToolarge {
                hiding_poly_degree,
                num_powers,
            })
        } else {
            Ok(())
        }
    }
    /*
    pub(crate) fn check_degrees_and_bounds<'a>(
        supported_degree: usize,
        max_degree: usize,
        enforced_degree_bounds: Option<&[usize]>,
        p: &'a LabeledPolynomial<E::Fr, P>,
    ) -> Result<(), Error> {
        if let Some(bound) = p.degree_bound() {
            let enforced_degree_bounds =
                enforced_degree_bounds.ok_or(Error::UnsupportedDegreeBound(bound))?;

            if enforced_degree_bounds.binary_search(&bound).is_err() {
                Err(Error::UnsupportedDegreeBound(bound))
            } else if bound < p.degree() || bound > max_degree {
                return Err(Error::IncorrectDegreeBound {
                    poly_degree: p.degree(),
                    degree_bound: p.degree_bound().unwrap(),
                    supported_degree,
                    label: p.label().to_string(),
                });
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }
    */
}

fn skip_leading_zeros_and_convert_to_bigints<F: PrimeField, P: UVPolynomial<F>>(
    p: &P,
) -> (usize, Vec<F::BigInt>) {
    let mut num_leading_zeros = 0;
    while num_leading_zeros < p.coeffs().len() && p.coeffs()[num_leading_zeros].is_zero() {
        num_leading_zeros += 1;
    }
    let coeffs = convert_to_bigints(&p.coeffs()[num_leading_zeros..]);
    (num_leading_zeros, coeffs)
}

fn convert_to_bigints<F: PrimeField>(p: &[F]) -> Vec<F::BigInt> {
    let to_bigint_time = start_timer!(|| "Converting polynomial coeffs to bigints");
    let coeffs = ark_std::cfg_iter!(p)
        .map(|s| s.into_repr())
        .collect::<Vec<_>>();
    end_timer!(to_bigint_time);
    coeffs
}

fn fft<G: Group + Group<ScalarField = F>, F: PrimeField>(v: &[G], omega: F) -> Vec<G>
{
    if v.len() == 2 {
        vec![
            v[0] + v[1].mul(&omega),
            v[0] + v[1]
        ]
    } else {
        let d2 = v.len() / 2;
        let mut evens = Vec::with_capacity(d2);
        let mut odds = Vec::with_capacity(d2);

        for (i, x) in v.iter().enumerate() {
            if i % 2 == 0 {
                evens.push(*x);
            } else {
                odds.push(*x);
            }
        }

        let o2 = omega * omega;
        let (evens, odds) = (fft(&evens, o2), fft(&odds, o2));

        let mut res = Vec::with_capacity(v.len());
        let mut acc = omega;
        for i in 0..v.len() {
            res.push(evens[i%d2] + odds[i%d2].mul(&acc));
            acc = acc.mul(&omega);
        }
    
        res
    }
}

/// Compute the vanishing polynomial in the input domain 
/// (which is not necessarily an FFT-friendly domain)
/// This is done by multiplying all the linear polynomials in a tree-like fashion
pub(crate) fn compute_vanishing_polynomial<F: PrimeField>(domain: &[F]) -> DensePoly<F> {
    //initialize poly1 with all the linear polynomials (X - h_i)
    let mut poly1 = Vec::new();
    for k in 0..domain.len() {
        poly1.push(DensePoly::<F>::from_coefficients_vec(vec![-domain[k], F::one()]));
    }
    //multiply all the linear polynomials in a tree fashion
    while poly1.len()>1 {
        let mut poly2 = Vec::<DensePoly<F>>::new();
        for j in 0..poly1.len()/2 {
            poly2.push(&poly1[2*j] * &poly1[2*j+1]);
        }
        poly1 = poly2.clone();
    }
    poly1[0].clone()
}
/*
pub(crate) fn fiat_shamir_hash<E: PairingEngine, P: UVPolynomial<E::Fr>>(
    name: &'static [u8],
    point: P::Point,
    com_p: &E::G1Affine,
    com_z: &E::G1Affine,
    com_q: &E::G1Affine,
    u: E::Fqk,
) -> E::Fr {
    let bytes = to_bytes![name, point, com_p, com_z, com_q, u].unwrap();
    let chg = Blake2s::digest(&bytes);
    let rho = E::Fr::from_random_bytes(&chg).unwrap();
    rho
}
*/

pub(crate) fn fiat_shamir_hashfn<E: PairingEngine, P: UVPolynomial<E::Fr>>(
    bytes: &[u8],
) -> E::Fr {
    //let bytes = to_bytes![name, point, com_p, com_z, com_q, u].unwrap();
    let chg = Blake2s::digest(bytes);

    let seed: [u8; 32] = FromBytes::read(&*chg).expect("failed to get [u32; 8]");
    let mut hrng = ChaChaRng::from_seed(seed);
    let rho = E::Fr::rand(&mut hrng);


    //let rho = E::Fr::from_random_bytes(&chg).unwrap();
    rho
}

#[cfg(test)]
mod tests {
    #![allow(non_camel_case_types)]
    use crate::kzg10::*;
    use crate::*;

    use ark_bls12_377::Bls12_377;
    use ark_bls12_381::Bls12_381;
    use ark_bls12_381::Fr;
    use ark_ec::PairingEngine;
    use ark_poly::univariate::DensePolynomial as DensePoly;
    use ark_std::test_rng;

    type UniPoly_381 = DensePoly<<Bls12_381 as PairingEngine>::Fr>;
    type UniPoly_377 = DensePoly<<Bls12_377 as PairingEngine>::Fr>;
    type KZG_Bls12_381 = HSNP_PC<Bls12_381, UniPoly_381>;

    impl<E: PairingEngine, P: UVPolynomial<E::Fr>> HSNP_PC<E, P> {
        /// Specializes the public parameters for a given maximum degree `d` for polynomials
        /// `d` should be less that `pp.max_degree()`.
        pub(crate) fn trim(
            pp: &UniversalParams<E>,
            mut supported_degree: usize,
        ) -> Result<(Powers<E>, VerifierKey<E>), Error> {
            if supported_degree == 1 {
                supported_degree += 1;
            }
            let powers_of_g = pp.powers_of_g[..=supported_degree].to_vec();
            let powers_of_gamma_g = (0..=supported_degree)
                .map(|i| pp.powers_of_gamma_g[&i])
                .collect();

            let powers = Powers {
                powers_of_g: ark_std::borrow::Cow::Owned(powers_of_g),
                powers_of_gamma_g: ark_std::borrow::Cow::Owned(powers_of_gamma_g),
            };
            let vk = VerifierKey {
                g: pp.powers_of_g[0],
                gamma_g: pp.powers_of_gamma_g[&0],
                h: pp.h,
                beta_h: pp.beta_h,
                prepared_h: pp.prepared_h.clone(),
                prepared_beta_h: pp.prepared_beta_h.clone(),
            };
            Ok((powers, vk))
        }
    }

    #[test]
    fn add_commitments_test() {
        let rng = &mut test_rng();
        let p = DensePoly::from_coefficients_slice(&[
            Fr::rand(rng),
            Fr::rand(rng),
            Fr::rand(rng),
            Fr::rand(rng),
            Fr::rand(rng),
        ]);
        let f = Fr::rand(rng);
        let mut f_p = DensePoly::zero();
        f_p += (f, &p);

        let degree = 4;
        let pp = KZG_Bls12_381::setup(degree, false, rng).unwrap();
        let (powers, _) = KZG_Bls12_381::trim(&pp, degree).unwrap();

        let hiding_bound = None;
        let (comm, _) = HSNP_PC::commit(&powers, &p, hiding_bound, Some(rng)).unwrap();
        let (f_comm, _) = HSNP_PC::commit(&powers, &f_p, hiding_bound, Some(rng)).unwrap();
        let mut f_comm_2 = Commitment::empty();
        f_comm_2 += (f, &comm);

        assert_eq!(f_comm, f_comm_2);
    }

    fn end_to_end_test_template<E, P>() -> Result<(), Error>
    where
        E: PairingEngine,
        P: UVPolynomial<E::Fr, Point = E::Fr>,
        for<'a, 'b> &'a P: Div<&'b P, Output = P>,
    {
        let rng = &mut test_rng();
        for _ in 0..100 {
            let mut degree = 0;
            while degree <= 1 {
                degree = usize::rand(rng) % 20;
            }
            let pp = HSNP_PC::<E, P>::setup(degree, false, rng)?;
            let (ck, vk) = HSNP_PC::<E, P>::trim(&pp, degree)?;
            let p = P::rand(degree, rng);
            let hiding_bound = Some(1);
            let (comm, rand) = HSNP_PC::<E, P>::commit(&ck, &p, hiding_bound, Some(rng))?;
            let point = E::Fr::rand(rng);
            let value = p.evaluate(&point);
            let proof = HSNP_PC::<E, P>::open(&ck, &p, point, &rand)?;
            assert!(
                HSNP_PC::<E, P>::check(&vk, &comm, point, value, &proof)?,
                "proof was incorrect for max_degree = {}, polynomial_degree = {}, hiding_bound = {:?}",
                degree,
                p.degree(),
                hiding_bound,
            );
        }
        Ok(())
    }

    fn linear_polynomial_test_template<E, P>() -> Result<(), Error>
    where
        E: PairingEngine,
        P: UVPolynomial<E::Fr, Point = E::Fr>,
        for<'a, 'b> &'a P: Div<&'b P, Output = P>,
    {
        let rng = &mut test_rng();
        for _ in 0..100 {
            let degree = 50;
            let pp = HSNP_PC::<E, P>::setup(degree, false, rng)?;
            let (ck, vk) = HSNP_PC::<E, P>::trim(&pp, 2)?;
            let p = P::rand(1, rng);
            let hiding_bound = Some(1);
            let (comm, rand) = HSNP_PC::<E, P>::commit(&ck, &p, hiding_bound, Some(rng))?;
            let point = E::Fr::rand(rng);
            let value = p.evaluate(&point);
            let proof = HSNP_PC::<E, P>::open(&ck, &p, point, &rand)?;
            assert!(
                HSNP_PC::<E, P>::check(&vk, &comm, point, value, &proof)?,
                "proof was incorrect for max_degree = {}, polynomial_degree = {}, hiding_bound = {:?}",
                degree,
                p.degree(),
                hiding_bound,
            );
        }
        Ok(())
    }

    fn batch_check_test_template<E, P>() -> Result<(), Error>
    where
        E: PairingEngine,
        P: UVPolynomial<E::Fr, Point = E::Fr>,
        for<'a, 'b> &'a P: Div<&'b P, Output = P>,
    {
        let rng = &mut test_rng();
        for _ in 0..10 {
            let mut degree = 0;
            while degree <= 1 {
                degree = usize::rand(rng) % 20;
            }
            let pp = HSNP_PC::<E, P>::setup(degree, false, rng)?;
            let (ck, vk) = HSNP_PC::<E, P>::trim(&pp, degree)?;
            let mut comms = Vec::new();
            let mut values = Vec::new();
            let mut points = Vec::new();
            let mut proofs = Vec::new();
            for _ in 0..10 {
                let p = P::rand(degree, rng);
                let hiding_bound = Some(1);
                let (comm, rand) = HSNP_PC::<E, P>::commit(&ck, &p, hiding_bound, Some(rng))?;
                let point = E::Fr::rand(rng);
                let value = p.evaluate(&point);
                let proof = HSNP_PC::<E, P>::open(&ck, &p, point, &rand)?;

                assert!(HSNP_PC::<E, P>::check(&vk, &comm, point, value, &proof)?);
                comms.push(comm);
                values.push(value);
                points.push(point);
                proofs.push(proof);
            }
            assert!(HSNP_PC::<E, P>::batch_check(
                &vk, &comms, &points, &values, &proofs, rng
            )?);
        }
        Ok(())
    }

    #[test]
    fn end_to_end_test() {
        end_to_end_test_template::<Bls12_377, UniPoly_377>().expect("test failed for bls12-377");
        end_to_end_test_template::<Bls12_381, UniPoly_381>().expect("test failed for bls12-381");
    }

    #[test]
    fn linear_polynomial_test() {
        linear_polynomial_test_template::<Bls12_377, UniPoly_377>()
            .expect("test failed for bls12-377");
        linear_polynomial_test_template::<Bls12_381, UniPoly_381>()
            .expect("test failed for bls12-381");
    }
    #[test]
    fn batch_check_test() {
        batch_check_test_template::<Bls12_377, UniPoly_377>().expect("test failed for bls12-377");
        batch_check_test_template::<Bls12_381, UniPoly_381>().expect("test failed for bls12-381");
    }

    #[test]
    fn test_degree_is_too_large() {
        let rng = &mut test_rng();

        let max_degree = 123;
        let pp = KZG_Bls12_381::setup(max_degree, false, rng).unwrap();
        let (powers, _) = KZG_Bls12_381::trim(&pp, max_degree).unwrap();

        let p = DensePoly::<Fr>::rand(max_degree + 1, rng);
        assert!(p.degree() > max_degree);
        assert!(KZG_Bls12_381::check_degree_is_too_large(p.degree(), powers.size()).is_err());
    }
}
