use ndarray::{s, Array, Array3, Axis};
use ndarray_rand::rand;
use ndarray_rand::rand::seq::{IteratorRandom, SliceRandom};
use ndarray_rand::rand_distr::{Distribution, Normal, Uniform};
use once_cell::sync::Lazy;
use rand::Rng;
use std::convert::TryInto;

//sample data
static NODE_DEMAND: Lazy<Vec<i32>> = Lazy::new(|| {
    vec![
        24, 8, 8, 20, 6, 8, 3, 12, 7, 5, 5, 4, 10, 6, 12, 15, 98, 90, 14, 8, 55, 9, 10, 6, 321, 33,
        5, 19, 31, 5, 15, 66, 10, 4, 7,
    ]
});

pub struct MolStructure {
    pub w: [Array3<i32>; 3],
    pub d: i32,
}
impl MolStructure {
    fn new_random(dims: [(usize, usize, usize); 3], demand: i32) -> MolStructure {
        assert!(dims[2].1 == NODE_DEMAND.len());
        let mut rng = rand::thread_rng();
        let mut con = Vec::<Array3<i32>>::new();
        // final layer has node wise demand requirements,
        // so assigned first. This way we get the demand for
        // each of the previous layer nodes, and can continue working backwards
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
        MolStructure {
            w: con.try_into().unwrap(),
            d: demand,
        }
    }
    fn copy(m: &MolStructure) -> MolStructure {
        MolStructure {
            w: m.w.clone(),
            d: m.d,
        }
    }
}

pub struct Molecule {
    pub w: MolStructure,
    pub pe: f64,
    pub ke: f64,
    pub num_hit: i32,
    pub min_w: MolStructure,
    pub min_pe: f64,
    pub min_hit: i32,
}

pub struct CRO {
    pub molecules: Vec<Molecule>,
    pub dims: [(usize, usize, usize); 3],
    pub buffer: f64,
    pub best_soln: MolStructure,
    pub best_pe: f64,
    pub objective_fn: fn(&MolStructure) -> f64,
    pub pop_size: i32,
    pub mole_coll: f64,
    pub ke_loss_rate: f64,
    pub alpha: i32,
    pub beta: f64,
    pub demand: i32,
}

impl CRO {
    pub fn new(
        pop_size: i32,
        potential: fn(&MolStructure) -> f64,
        dims: [(usize, usize, usize); 3],
        demand: i32,
    ) -> CRO {
        let mut mols: Vec<Molecule> = Vec::<Molecule>::new();
        let mut ovr_min: f64 = 1e9;
        let mut ovr_min_idx: usize = (pop_size + 1) as usize;
        for i in 0..pop_size {
            let w_: MolStructure = MolStructure::new_random(dims, demand);
            let w_copy: MolStructure = MolStructure::copy(&w_);
            let pe_: f64 = potential(&w_);
            mols.push(Molecule {
                w: w_,
                pe: pe_,
                ke: 400.0, // 0 to 500
                num_hit: 0,
                min_w: w_copy,
                min_pe: pe_,
                min_hit: 0,
            });
            if pe_ < ovr_min {
                ovr_min = pe_;
                ovr_min_idx = i as usize;
            }
        }
        let best_w: MolStructure = MolStructure::copy(&mols[ovr_min_idx].w);
        // edit hyperparameters to suit problem
        CRO {
            molecules: mols,
            dims,
            buffer: 2300.0, // 0 to 2500
            best_soln: best_w,
            best_pe: ovr_min,
            objective_fn: potential,
            pop_size,
            mole_coll: 0.7,     // 0 to 1
            ke_loss_rate: 0.25, // 0 to 1
            alpha: 400,         // 0 to 1000
            beta: 40.0,         // 0 to 100
            demand,
        }
    }
    fn get_neighbourhood(&self, a: &MolStructure) -> MolStructure {
        let mut rng = rand::thread_rng();
        let mut a_: MolStructure = MolStructure::copy(a);
        // two opt exchanging
        // cannot affect nodewise supply in last layer
        // hence j fixed
        for j in 0..self.dims[2].1 {
            let i1 = rng.gen_range(0..a.w[2].raw_dim()[0]);
            let k1 = rng.gen_range(0..a.w[2].raw_dim()[2]);
            let i2 = rng.gen_range(0..a.w[2].raw_dim()[0]);
            let k2 = rng.gen_range(0..a.w[2].raw_dim()[2]);
            let temp = a_.w[2][[i1, j, k1]];
            a_.w[2][[i1, j, k1]] = a_.w[2][[i2, j, k2]];
            a_.w[2][[i2, j, k2]] = temp;
        }
        // gaussian mutation of last layer
        let gaussian = Normal::new(0., 5.).unwrap();
        for j in 0..self.dims[2].2 {
            let mut v_delta: Vec<i32> = Vec::<i32>::new();
            let mut half: usize = a.w[2].raw_dim()[0] * a.w[2].raw_dim()[2];
            let is_odd: bool = (half % 2) != 0;
            half /= 2;
            for _ in 0..half {
                let delta: f64 = gaussian.sample(&mut rng);
                let delta_int: i32 = delta.round() as i32;
                v_delta.push(delta_int);
                v_delta.push(-delta_int);
            }
            if is_odd {
                v_delta.push(0);
            }
            v_delta.shuffle(&mut rng);
            let m_delta =
                Array::from_shape_vec((a.w[2].raw_dim()[0], a.w[2].raw_dim()[2]), v_delta).unwrap();
            let mut slice = a_.w[2].slice_mut(s![.., j, ..]);
            slice += &m_delta;
            // reflect negative values
            slice.mapv_inplace(|y| y.abs());
            if slice.sum() != NODE_DEMAND[j] {
                let mut m = slice.mapv(|y: i32| f64::from(y));
                m = &m * ((NODE_DEMAND[j] as f64) / (slice.sum() as f64));
                let mut rnc: f64 = 0.0;
                for i in 0..m.shape()[0] {
                    for j in 0..m.shape()[1] {
                        m[[i, j]] += rnc;
                        rnc = m[[i, j]] - m[[i, j]].round();
                        m[[i, j]] = m[[i, j]].round();
                    }
                }
                slice.assign(&m.mapv(|y: f64| y as i32));
            }
        }
        // work backwards assigning new routes randomly as per node demand
        for x in (0..2).rev() {
            let mut l: Array3<i32> = Array3::zeros(self.dims[x]);
            let int_demand = a_.w[x + 1].sum_axis(Axis(2)).sum_axis(Axis(1));
            for j in 0..self.dims[x].1 {
                let prod: usize = self.dims[x].0 * self.dims[x].2;
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
                    let arr = Array::from_shape_vec((self.dims[x].0, self.dims[x].2), r).unwrap();
                    l.slice_mut(s![.., j, ..]).assign(&arr);
                } else {
                    let r: Vec<i32> = vec![int_demand[j]];
                    let arr = Array::from_shape_vec((self.dims[x].0, self.dims[x].2), r).unwrap();
                    l.slice_mut(s![.., j, ..]).assign(&arr);
                }
            }
            a_.w[x].assign(&l);
        }
        a_
    }
    fn on_wall_ineffective_collision(&mut self, z: usize) {
        let w_: MolStructure = self.get_neighbourhood(&self.molecules[z].w);
        let pe_: f64 = (self.objective_fn)(&w_);
        let m: &mut Molecule = &mut self.molecules[z];
        m.num_hit += 1;
        if (m.ke + m.pe) > pe_ {
            let a: f64 = Uniform::new(self.ke_loss_rate, 1.).sample(&mut rand::thread_rng());
            let rem: f64 = m.ke + m.pe - pe_;
            let ke_: f64 = rem * a;
            self.buffer += rem - ke_;
            m.pe = pe_;
            m.ke = ke_;
            m.w = w_;
            if m.pe < m.min_pe {
                m.min_w = MolStructure::copy(&m.w);
                m.min_pe = m.pe;
                m.min_hit = m.num_hit;
            }
        }
    }
    fn intermolecular_ineffective_collision(&mut self, y: usize, z: usize) {
        assert!(y != z);
        let w_1: MolStructure = self.get_neighbourhood(&self.molecules[y].w);
        let w_2: MolStructure = self.get_neighbourhood(&self.molecules[z].w);
        let m: &mut Vec<Molecule> = &mut self.molecules;
        let pe_1: f64 = (self.objective_fn)(&w_1);
        let pe_2: f64 = (self.objective_fn)(&w_2);
        m[y].num_hit += 1;
        m[z].num_hit += 1;
        if (m[y].pe + m[y].ke + m[z].pe + m[z].ke) > (pe_1 + pe_2) {
            let e_inter: f64 = m[y].pe + m[z].ke + m[y].pe + m[z].ke - (pe_1 + pe_2);
            let a: f64 = Uniform::new(0., 1.).sample(&mut rand::thread_rng());
            let ke_1: f64 = a * e_inter;
            let ke_2: f64 = e_inter - ke_1;
            m[y].pe = pe_1;
            m[y].ke = ke_1;
            m[y].w = w_1;
            m[z].pe = pe_2;
            m[z].ke = ke_2;
            m[z].w = w_2;
            if m[y].pe < m[y].min_pe {
                m[y].min_w = MolStructure::copy(&m[y].w);
                m[y].min_pe = m[y].pe;
                m[y].min_hit = m[y].num_hit;
            }
            if m[z].pe < m[z].min_pe {
                m[z].min_w = MolStructure::copy(&m[z].w);
                m[z].min_pe = m[z].pe;
                m[z].min_hit = m[z].num_hit;
            }
        }
    }
    fn decomposition(&mut self, z: usize) {
        let delta1: f64 = Uniform::new(0., 1.).sample(&mut rand::thread_rng());
        let delta2: f64 = Uniform::new(0., 1.).sample(&mut rand::thread_rng());
        let w_1: MolStructure = self.get_neighbourhood(&self.molecules[z].w);
        let w_2: MolStructure = self.get_neighbourhood(&self.molecules[z].w);
        let pe_1: f64 = (self.objective_fn)(&w_1);
        let pe_2: f64 = (self.objective_fn)(&w_2);
        let m: &mut Molecule = &mut self.molecules[z];
        if (m.pe + m.ke) > (pe_1 + pe_2) {
            let e_dec: f64 = (m.pe + m.ke) - (pe_1 + pe_2);
            let delta3: f64 = Uniform::new(0., 1.).sample(&mut rand::thread_rng());
            let ke_1: f64 = delta3 * e_dec;
            let ke_2: f64 = e_dec - ke_1;
            //creating 2 new molecules
            let w_1_copy: MolStructure = MolStructure::copy(&w_1);
            let w_2_copy: MolStructure = MolStructure::copy(&w_2);
            let m1: Molecule = Molecule {
                w: w_1,
                pe: pe_1,
                ke: ke_1,
                num_hit: 0,
                min_w: w_1_copy,
                min_pe: pe_1,
                min_hit: 0,
            };
            let m2: Molecule = Molecule {
                w: w_2,
                pe: pe_2,
                ke: ke_2,
                num_hit: 0,
                min_w: w_2_copy,
                min_pe: pe_2,
                min_hit: 0,
            };
            self.molecules.push(m1);
            self.molecules.push(m2);
            self.molecules.swap_remove(z);
        } else if (m.pe + m.ke + delta1 * delta2 * self.buffer) > (pe_1 + pe_2) {
            let e_dec: f64 = (m.pe + m.ke + delta1 * delta2 * self.buffer) - (pe_1 + pe_2);
            let delta3: f64 = Uniform::new(0., 1.).sample(&mut rand::thread_rng());
            let ke_1: f64 = delta3 * e_dec;
            let ke_2: f64 = e_dec - ke_1;
            self.buffer = (1. - delta1 * delta2) * self.buffer;
            //creating 2 new molecules
            let w_1_copy: MolStructure = MolStructure::copy(&w_1);
            let w_2_copy: MolStructure = MolStructure::copy(&w_2);
            let m1: Molecule = Molecule {
                w: w_1,
                pe: pe_1,
                ke: ke_1,
                num_hit: 0,
                min_w: w_1_copy,
                min_pe: pe_1,
                min_hit: 0,
            };
            let m2: Molecule = Molecule {
                w: w_2,
                pe: pe_2,
                ke: ke_2,
                num_hit: 0,
                min_w: w_2_copy,
                min_pe: pe_2,
                min_hit: 0,
            };
            self.molecules.push(m1);
            self.molecules.push(m2);
            self.molecules.swap_remove(z);
        } else {
            m.num_hit += 1;
        }
    }
    fn synthesis(&mut self, y: usize, z: usize) {
        assert!(y != z);
        let m1: &Molecule = &self.molecules[y];
        let m2: &Molecule = &self.molecules[z];
        let mut w_: MolStructure = MolStructure::copy(&m2.w);
        //combining two molecules
        let gen = Uniform::new(0, 2);
        let mut rng = rand::thread_rng();
        for j in 0..w_.w[2].raw_dim()[1] {
            let mut slice = w_.w[2].slice_mut(s![.., j, ..]);
            if gen.sample(&mut rng) == 0 {
                slice.assign(&m1.w.w[2].slice(s![.., j, ..]));
            }
        }
        for x in (0..2).rev() {
            let mut l: Array3<i32> = Array3::zeros(self.dims[x]);
            let int_demand = w_.w[x + 1].sum_axis(Axis(2)).sum_axis(Axis(1));
            for j in 0..self.dims[x].1 {
                let prod: usize = self.dims[x].0 * self.dims[x].2;
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
                    let arr = Array::from_shape_vec((self.dims[x].0, self.dims[x].2), r).unwrap();
                    l.slice_mut(s![.., j, ..]).assign(&arr);
                } else {
                    let r: Vec<i32> = vec![int_demand[j]];
                    let arr = Array::from_shape_vec((self.dims[x].0, self.dims[x].2), r).unwrap();
                    l.slice_mut(s![.., j, ..]).assign(&arr);
                }
            }
            w_.w[x].assign(&l);
        }
        let pe_: f64 = (self.objective_fn)(&w_);
        if (m1.pe + m1.ke + m2.pe + m2.ke) > pe_ {
            let ke_ = (m1.pe + m1.ke + m2.pe + m2.ke) - pe_;
            let w_copy = MolStructure::copy(&w_);
            self.molecules[y] = Molecule {
                w: w_,
                pe: pe_,
                ke: ke_,
                num_hit: 0,
                min_w: w_copy,
                min_pe: pe_,
                min_hit: 0,
            };
            self.molecules.swap_remove(z);
        } else {
            self.molecules[y].num_hit += 1;
            self.molecules[z].num_hit += 1;
        }
    }
    pub fn main_cycle(&mut self) {
        let mut rng = rand::thread_rng();
        let b: f64 = Uniform::new(0., 1.).sample(&mut rng);
        if b > self.mole_coll || self.molecules.len() < 2 {
            let i: usize = rng.gen_range(0..self.molecules.len());
            if self.molecules[i].num_hit - self.molecules[i].min_hit > self.alpha {
                self.decomposition(i);
            } else {
                self.on_wall_ineffective_collision(i);
            }
        } else {
            let idx: Vec<usize> = (0..self.molecules.len()).choose_multiple(&mut rng, 2);
            if self.molecules[idx[0]].ke < self.beta && self.molecules[idx[1]].ke < self.beta {
                self.synthesis(idx[0], idx[1]);
            } else {
                self.intermolecular_ineffective_collision(idx[0], idx[1]);
            }
        }
        //update best soln and best pe
        for m in &self.molecules {
            if m.min_pe < self.best_pe {
                self.best_pe = m.min_pe;
                self.best_soln = MolStructure::copy(&m.min_w);
            }
        }
    }
}
