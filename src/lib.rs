//! # lpc-rs
//!
//! `lpc-rs` is a library for calculating Linear Predictive Coding (LPC) coefficients.
//! It provides three methods to calculate LPC coefficients.
//! - Low speed method (Temporarily commented out due to suspension of updates to dependent libraries)
//! - High speed method
//! - Burg method

use ndarray::prelude::*;
// use ndarray_inverse::Inverse;

/// find the correlation of the input array
/// # Arguments
/// [;N]
///
/// # Returns
/// [;N]
pub fn correlate(a: ArrayView1<f64>) -> Array1<f64> {
    a.iter()
        .enumerate()
        .map(|(n, _)| {
            a.slice(s![n..])
                .iter()
                .zip(a.iter())
                .map(|(x, y)| x * y)
                .sum::<f64>()
        })
        .collect()
}

// pub fn calc_lpc_by_low_speed(a: ArrayView1<f64>, depth: usize) -> Array1<f64> {
//     let r = correlate(a);

//     let mut large_r: Array2<f64> = Array2::zeros((depth, depth));
//     for i in 0..depth {
//         for j in 0..depth {
//             large_r[[i, j]] = r[(i as isize - j as isize).abs() as usize];
//         }
//     }

//     println!("{:?}", large_r);

//     let r = r.slice(s![1..=depth]);

//     println!("{:?}", r);

//     let inverse_large_r = large_r.inv().unwrap();

//     let a = inverse_large_r.dot(&r);

//     let minus_a = a.mapv(|x| -x);

//     minus_a
// }

/// https://qiita.com/hirokisince1998/items/fd50c0515c7788458fce
/// Levinson-Durbin recursion
pub fn calc_lpc_by_levinson_durbin(a: ArrayView1<f64>, depth: usize) -> Option<Array1<f64>> {
    if a.len() < depth {
        return None;
    }

    let r = correlate(a);
    println!("{:?}", r);
    let r = r.slice(s![..=depth]);
    println!("{:?}", r);

    fn calc_lpc_by_high_speed_inner(
        a: ArrayView1<f64>,
        depth: usize,
        r: ArrayView1<f64>,
    ) -> (Array1<f64>, f64) {
        if depth == 1 {
            let a = Array1::from_iter(vec![1.0, -r[1] as f64 / r[0]]);
            let e = a.dot(&r.slice(s![..2]));

            // println!("{:?}", a);

            (a, e)
        } else {
            let (aa, ee) = calc_lpc_by_high_speed_inner(a, depth - 1, r);

            let kk = -aa.dot(&r.slice(s![1..=depth; -1])) / ee;

            let large_u = ndarray::concatenate![Axis(0), aa.view(), Array1::from_elem(1, 0.0)];

            let large_v = large_u.slice(s![..; -1]);

            let a = large_u.clone() + large_v.mapv(|x| x * kk);

            let e = ee * (1.0 - kk * kk);

            (a, e)
        }
    }

    let (a, _) = calc_lpc_by_high_speed_inner(a, depth, r.view());

    Some(a.slice_move(s![1..]))
}

/// Burg method
pub fn calc_lpc_by_burg(x: ArrayView1<f64>, depth: usize) -> Option<Array1<f64>> {
    if x.len() < depth {
        return None;
    }

    let mut a = Array1::<f64>::zeros(depth + 1);

    let mut k = Array1::<f64>::zeros(depth);

    a[0] = 1.0;

    let mut f = x.to_owned();

    let mut b = x.to_owned();

    let n = x.len();

    for p in 0..depth {
        let kf = f.slice(s![p + 1..]);
        let kb = b.slice(s![..n - p - 1]);
        // element-wise sum of squares
        let d = kf.iter().map(|x| x * x).sum::<f64>() + kb.iter().map(|x| x * x).sum::<f64>();
        k[p] = -2.0 * kf.iter().zip(kb.iter()).map(|(x, y)| x * y).sum::<f64>() / d;
        let u = a.slice(s![..=p + 1]);
        let v = u.slice(s![..; -1]);

        println!("u: {:?}", u);
        println!("v: {:?}", v);

        let added = &u + &v.mapv(|x| x * k[p]);
        a.slice_mut(s![..=p + 1]).assign(&added.view());
        let fu = b.slice(s![..n - p - 1]).mapv(|x| x * k[p]);
        let bu = f.slice(s![p + 1..]).mapv(|x| x * k[p]);
        f.slice_mut(s![p + 1..])
            .iter_mut()
            .zip(fu.iter())
            .for_each(|(x, fu)| *x += *fu);
        b.slice_mut(s![..n - p - 1])
            .iter_mut()
            .zip(bu.iter())
            .for_each(|(x, bu)| *x += bu);
    }

    Some(a.slice_move(s![1..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlate() {
        let a = Array1::from(vec![2., 3., -1., -2., 1., 4., 1.]);
        let expected = Array1::from(vec![36.0, 11.0, -16.0, -7.0, 13.0, 11.0, 2.0]);
        assert_eq!(correlate(a.view()), expected);
    }

    // #[test]
    // fn test_calc_lpc_by_low_speed() {
    //     let a = Array1::from(vec![2., 3., -1., -2., 1., 4., 1.]);
    //     let depth = 3;
    //     let expected = Array1::from(vec![
    //         -0.6919053749597682,
    //         0.7615062761506275,
    //         -0.34575152880592214,
    //     ]);
    //     assert_eq!(calc_lpc_by_low_speed(a.view(), depth), expected);
    // }

    #[test]
    fn test_calc_lpc_by_high_speed() {
        let a = Array1::from(vec![2., 3., -1., -2., 1., 4., 1.]);
        let depth = 3;
        let expected = Array1::from(vec![
            -0.6919053749597684,
            0.7615062761506278,
            -0.3457515288059223,
        ]);
        assert_eq!(calc_lpc_by_levinson_durbin(a.view(), depth), Some(expected));
    }

    #[test]
    fn test_calc_lpc_by_burg() {
        let a = Array1::from(vec![2., 3., -1., -2., 1., 4., 1.]);
        let depth = 3;
        let expected = Array1::from(vec![
            -1.0650404360323664,
            1.157238171254371,
            -0.5771692748969812,
        ]);
        assert_eq!(calc_lpc_by_burg(a.view(), depth), Some(expected));
    }
}
