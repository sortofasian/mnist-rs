use rayon::prelude::*;

pub trait Function {
    fn f(&self, x: &Vec<f32>) -> Vec<f32>;
    fn df(&self, x: &Vec<f32>) -> Vec<f32>;
}

pub trait Cost {
    fn f(&self, yp: &Vec<f32>, y: &Vec<f32>) -> f32;
    fn df(&self, yp: &Vec<f32>, y: &Vec<f32>) -> Vec<f32>;
}

pub struct Crossent;
impl Cost for Crossent {
    fn f(&self, yp: &Vec<f32>, y: &Vec<f32>) -> f32 {
        yp.par_iter().zip(y).map(|(yp, y)| -yp.ln() * y).sum()
    }
    fn df(&self, yp: &Vec<f32>, y: &Vec<f32>) -> Vec<f32> {
        yp.par_iter().zip(y).map(|(yp, y)| -y / yp).collect() //shouldn't de/dy_p = -y/y_p???
    }
}

pub struct Relu;
impl Function for Relu {
    fn f(&self, x: &Vec<f32>) -> Vec<f32> {
        x.par_iter().map(|x| x.max(0.0)).collect()
    }
    fn df(&self, x: &Vec<f32>) -> Vec<f32> {
        let x = self.f(&x);
        x.into_par_iter()
            .map(|x| match 0.0 < x {
                true => 1.0,
                false => 0.0,
            })
            .collect()
    }
}

pub struct Softmax;
impl Function for Softmax {
    fn f(&self, x: &Vec<f32>) -> Vec<f32> {
        let max = x.par_iter().fold(||f32::MIN, |a, b| a.max(*b)).collect::<Vec<f32>>()[0];
        let x = x.into_par_iter().map(|x| (x - max).exp());
        let sum: f32 = x.clone().sum();

        x.map(|xn| xn / sum).collect()
    }
    fn df(&self, x: &Vec<f32>) -> Vec<f32> {
        //let max = x.par_iter().fold(||f32::MIN, |a, b| a.max(*b)).collect::<Vec<f32>>()[0];
        //let x = x.par_iter().map(|x| (x - max).exp());
        //let sum: f32 = x.clone().sum();

        //x.map(|xn| xn / sum).map(|s| s * (1.0 - s)).collect();
        x.to_vec()
    }
}
