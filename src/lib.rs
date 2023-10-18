use ndarray::{s, Array, Array3, Axis};
use ndarray_rand::rand;
use ndarray_rand::rand::seq::IteratorRandom;
use ndarray_rand::rand_distr::{Distribution, Uniform};
use ndarray_rand::RandomExt;
use once_cell::sync::Lazy;
use std::convert::TryInto;

const THRESHOLD: f64 = 50.0;

//sample data
static NODE_DEMAND: Lazy<Vec<i32>> = Lazy::new(|| {
    vec![
        24, 8, 8, 20, 6, 8, 3, 12, 7, 5, 5, 4, 10, 6, 12, 15, 98, 90, 14, 8, 55, 9, 10, 6, 321, 33,
        5, 19, 31, 5, 15, 66, 10, 4, 7,
    ]
});

// static NODE_DEMAND: Lazy<Vec<i32>> = Lazy::new(|| {
//     vec![
//         5, 1, 2, 4, 2, 2, 1, 3, 30, 2, 30, 0, 2, 1, 1, 1, 29, 6, 1, 1, 12, 1, 121, 3, 12, 3, 0, 2,
//         3, 1, 5, 236, 2, 1, 3,
//     ]
// });

// static NODE_DEMAND: Lazy<Vec<i32>> = Lazy::new(|| {
//     vec![
//         68, 11, 17, 28, 23, 24, 2, 35, 7, 13, 15, 6, 31, 17, 12, 35, 166, 140, 25, 9, 74, 14, 28, 17, 248, 34, 6, 21,
//         28, 7, 30, 118, 19, 8, 19,
//     ]
// });

pub struct Info {
    pub pos: [Array3<i32>; 3],
    pub vel: Array3<f64>,
    pub d: i32,
}

impl Info {
    fn new_random(dims: [(usize, usize, usize); 3], demand: i32) -> Info {
        assert!(dims[2].1 == NODE_DEMAND.len());
        let mut rng = rand::thread_rng();
        let mut con = Vec::<Array3<i32>>::new();
        let mut fl: Array3<i32> = Array3::zeros(dims[2]);
        for j in 0..dims[2].1 {
            let prod: usize = dims[2].0 * dims[2].2;
            let cnt = NODE_DEMAND[j] + (prod as i32);
            let mut vals: Vec<i32> = (1..cnt).choose_multiple(&mut rng, prod - 1);
            vals.sort_unstable();
            let mut r: Vec<i32> = vec![0; prod];
            r[0] = vals[0] - 1;
            for i in 1..prod - 1 {
                r[i] = vals[i] - vals[i - 1] - 1;
            }
            r[prod - 1] = cnt - vals[prod - 2] - 1;
            let arr = Array::from_shape_vec((dims[2].0, dims[2].2), r).unwrap();
            fl.slice_mut(s![.., j, ..]).assign(&arr);
        }
        con.push(fl);
        for x in (0..2).rev() {
            let mut l: Array3<i32> = Array3::zeros(dims[x]);
            let int_demand = con.last().unwrap().sum_axis(Axis(2)).sum_axis(Axis(1));
            for j in 0..dims[x].1 {
                let prod: usize = dims[x].0 * dims[x].2;
                if prod != 1 {
                    let cnt = int_demand[j] + (prod as i32);
                    let mut vals: Vec<i32> = (1..cnt).choose_multiple(&mut rng, prod - 1);
                    vals.sort_unstable();
                    let mut r: Vec<i32> = vec![0; prod];
                    r[0] = vals[0] - 1;
                    for i in 1..prod - 1 {
                        r[i] = vals[i] - vals[i - 1] - 1;
                    }
                    r[prod - 1] = cnt - vals[prod - 2] - 1;
                    let arr = Array::from_shape_vec((dims[x].0, dims[x].2), r).unwrap();
                    l.slice_mut(s![.., j, ..]).assign(&arr);
                } else {
                    let r: Vec<i32> = vec![int_demand[j]];
                    let arr = Array::from_shape_vec((dims[x].0, dims[x].2), r).unwrap();
                    l.slice_mut(s![.., j, ..]).assign(&arr);
                }
            }
            con.push(l);
        }
        con.reverse();
        let vel = Array::random(dims[2], Uniform::new(-1., 1.));
        Info {
            pos: con.try_into().unwrap(),
            vel,
            d: demand,
        }
    }
    fn copy(p: &Info) -> Info {
        Info {
            pos: p.pos.clone(),
            vel: p.vel.clone(),
            d: p.d,
        }
    }
}

pub struct Particle {
    pub x: Info,
    pub f: f64,
    pub best_known: Info,
    pub min_f: f64,
}

pub struct PSO {
    pub particles: Vec<Particle>,
    pub dims: [(usize, usize, usize); 3],
    pub g_best_known: Info,
    pub best_f: f64,
    pub w: f64,
    pub phi_g: f64,
    pub phi_p: f64,
    pub objective_fn: fn(&Info) -> f64,
    pub swarm_size: i32,
    pub demand: i32,
}
impl PSO {
    pub fn new(
        swarm_size: i32,
        obj_f: fn(&Info) -> f64,
        dims: [(usize, usize, usize); 3],
        demand: i32,
    ) -> PSO {
        let mut pars: Vec<Particle> = Vec::new();
        let mut ovr_min: f64 = 1e9;
        let mut ovr_min_idx: usize = (swarm_size + 1) as usize;
        for i in 0..swarm_size {
            let x: Info = Info::new_random(dims, demand);
            let x_copy: Info = Info::copy(&x);
            let f_val: f64 = obj_f(&x);
            pars.push(Particle {
                x,
                f: f_val,
                best_known: x_copy,
                min_f: f_val,
            });
            if f_val < ovr_min {
                ovr_min = f_val;
                ovr_min_idx = i as usize;
            }
        }
        let g_best_known: Info = Info::copy(&pars[ovr_min_idx].x);
        PSO {
            particles: pars,
            g_best_known,
            dims,
            best_f: ovr_min,
            w: 0.9,
            phi_g: 1.3,
            phi_p: 1.3,
            objective_fn: obj_f,
            swarm_size,
            demand,
        }
    }
    fn change_velocity(&mut self, idx: usize) {
        let p: &mut Particle = &mut self.particles[idx];
        p.x.vel = Array::random(self.dims[2], Uniform::new(-1., 1.));
    }
    fn update_velocity(&mut self, idx: usize) {
        let p: &mut Particle = &mut self.particles[idx];
        let u: Uniform<f64> = Uniform::new(0., 2.);
        let mut rng = rand::thread_rng();
        let r1: f64 = u.sample(&mut rng);
        let r2: f64 = u.sample(&mut rng);
        p.x.vel = self.w * &p.x.vel;
        p.x.vel =
            &p.x.vel + self.phi_p * r1 * (&p.best_known.pos[2] - &p.x.pos[2]).mapv(|x| x as f64);
        p.x.vel = &p.x.vel
            + self.phi_g * r2 * (&self.g_best_known.pos[2] - &p.x.pos[2]).mapv(|x| x as f64);
        p.x.vel.mapv_inplace(|x: f64| x.clamp(-5., 5.));
    }
    fn update_position(&mut self, idx: usize) {
        let p: &mut Particle = &mut self.particles[idx];
        for j in 0..self.dims[2].1 {
            let mut slice = p.x.pos[2].slice_mut(s![.., j, ..]);
            slice += &p.x.vel.slice(s![.., j, ..]).mapv(|x| x.round() as i32);
            slice.mapv_inplace(|x: i32| x.abs());
            let mut m = slice.mapv(|y: i32| f64::from(y));
            if slice.sum() == 0 {
                let new_ = Array::random(slice.raw_dim(), Uniform::new(0.,1.));
                m.assign(&new_);
            }
            m = ((NODE_DEMAND[j] as f64) / (m.sum() as f64)) * &m;
            let mut rnc: f64 = 0.0;
            for i in 0..m.shape()[0] {
                for j in 0..m.shape()[1] {
                    m[[i, j]] += rnc;
                    rnc = m[[i, j]] - m[[i, j]].round();
                    m[[i, j]] = m[[i, j]].round();
                }
            }
            slice.assign(&m.mapv(|y: f64| y as i32));
            assert!(slice.sum()==NODE_DEMAND[j]);
        }
        for i in (0..2).rev() {
            let int_demand = p.x.pos[i + 1].sum_axis(Axis(2)).sum_axis(Axis(1));
            for j in 0..self.dims[i].1 {
                let mut slice = p.x.pos[i].slice_mut(s![.., j, ..]);
                let mut m = slice.mapv(|y: i32| f64::from(y));
                m = &m * ((int_demand[j] as f64) / (slice.sum() as f64));
                let mut rnc: f64 = 0.0;
                for i in 0..m.shape()[0] {
                    for j in 0..m.shape()[1] {
                        m[[i, j]] += rnc;
                        rnc = m[[i, j]] - m[[i, j]].round();
                        m[[i, j]] = m[[i, j]].round();
                    }
                }
                slice.assign(&m.mapv(|y: f64| y as i32));
                assert!(slice.sum()==int_demand[j])
            }
        }
    }
    pub fn update_particle(&mut self, idx: usize) {
        self.update_velocity(idx);
        self.update_position(idx);
        if (self.objective_fn)(&self.particles[idx].x) < self.particles[idx].min_f {
            self.particles[idx].min_f = (self.objective_fn)(&self.particles[idx].x);
            self.particles[idx].best_known = Info::copy(&self.particles[idx].x);
        }
    }
    fn get_inertia(t: i32) -> f64 {
        let k: f64 = 0.15 * (t as f64);
        1.4 * (1. - 1. / (1. + k.exp())) + 0.2
    }
    pub fn main_cycle(&mut self, t: i32) {
        for idx in 0..self.particles.len() {
            self.update_particle(idx);
            let new_f: f64 = (self.objective_fn)(&self.particles[idx].x);
            if new_f < self.best_f {
                self.best_f = new_f;
                self.g_best_known = Info::copy(&self.particles[idx].x);
            } else if new_f > self.particles[idx].min_f + THRESHOLD {
                // if particle roams too far off
                self.particles[idx].x = Info::copy(&self.particles[idx].best_known);
                self.change_velocity(idx);
            }
        }
        self.w = Self::get_inertia(t);
    }
}
