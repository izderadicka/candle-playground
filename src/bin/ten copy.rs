use anyhow::{Result, bail};
use candle_core::{DType, Device, Tensor};

fn main() -> Result<()> {
    let device = Device::Cpu;

    // Create a square matrix
    let matrix = Tensor::new(&[[4f32, 2., 1.], [2., 5., 3.], [1., 3., 6.]], &device)?;
    println!("Matrix: {}", matrix);

    let (n, m) = matrix.dims2()?;
    if n != m {
        bail!("expected a square matrix, got {n}x{m}");
    }

    // Compute the trace (sum of diagonal elements).
    // Candle has no `trace`, but masking with the identity and summing does it.
    let eye = Tensor::eye(n, DType::F32, &device)?;
    let trace = matrix.mul(&eye)?.sum_all()?;
    println!("Trace: {}", trace);

    // Candle has no determinant or inverse either. One Gauss-Jordan pass
    // with partial pivoting yields both at once.
    let (inverse, determinant) = inverse_and_det(&matrix)?;
    println!("Determinant: {}", determinant);
    println!("Inverse: {}", inverse);

    // Compute eigenvalues and eigenvectors (symmetric matrices only)
    let (eigenvalues, eigenvectors) = eig_symmetric(&matrix)?;
    println!("Eigenvalues: {}", eigenvalues);
    println!("Eigenvectors (columns): {}", eigenvectors);

    // Compute the norm
    let norm = matrix.flatten_all()?.sqr()?.sum_all()?.sqrt()?;
    println!("Frobenius norm: {}", norm);

    // Sanity check: M * M^-1 should be the identity
    println!("M * M^-1: {}", matrix.matmul(&inverse)?);

    Ok(())
}

/// Gauss-Jordan elimination with partial pivoting.
/// Returns the inverse and the determinant of a square matrix.
fn inverse_and_det(matrix: &Tensor) -> Result<(Tensor, f32)> {
    let (n, _) = matrix.dims2()?;
    let mut a = matrix.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    // Augment with the identity; it turns into the inverse as `a` becomes I.
    let mut inv = vec![vec![0f32; n]; n];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.;
    }

    let mut det = 1f32;
    for col in 0..n {
        // Pivot on the largest magnitude entry in the column for stability.
        let pivot = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()));
        let pivot = pivot.expect("column range is non-empty");
        if a[pivot][col] == 0. {
            bail!("matrix is singular, it has no inverse");
        }
        if pivot != col {
            a.swap(pivot, col);
            inv.swap(pivot, col);
            det = -det; // each row swap flips the sign
        }

        let d = a[col][col];
        det *= d;
        for k in 0..n {
            a[col][k] /= d;
            inv[col][k] /= d;
        }

        // Eliminate this column from every other row.
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor == 0. {
                continue;
            }
            for k in 0..n {
                a[row][k] -= factor * a[col][k];
                inv[row][k] -= factor * inv[col][k];
            }
        }
    }

    let flat: Vec<f32> = inv.into_iter().flatten().collect();
    Ok((Tensor::from_vec(flat, (n, n), matrix.device())?, det))
}

/// Cyclic Jacobi eigenvalue algorithm.
///
/// Only valid for *symmetric* matrices — a non-symmetric input silently
/// produces garbage, so symmetry is checked up front.
/// Returns the eigenvalues and a matrix whose *columns* are the eigenvectors.
fn eig_symmetric(matrix: &Tensor) -> Result<(Tensor, Tensor)> {
    const MAX_SWEEPS: usize = 100;
    const EPS: f32 = 1e-9;

    let (n, _) = matrix.dims2()?;
    let mut a = matrix.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    for i in 0..n {
        for j in 0..i {
            if (a[i][j] - a[j][i]).abs() > 1e-5 {
                bail!("eig_symmetric requires a symmetric matrix");
            }
        }
    }

    // Accumulates the rotations; ends up holding the eigenvectors as columns.
    let mut v = vec![vec![0f32; n]; n];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.;
    }

    for _ in 0..MAX_SWEEPS {
        // Stop once the off-diagonal mass is negligible.
        let off: f32 = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .filter(|(i, j)| i != j)
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        if off < EPS {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < EPS {
                    continue;
                }
                // Rotation angle that zeroes out a[p][q].
                let theta = (a[q][q] - a[p][p]) / (2. * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.).sqrt());
                let c = 1. / (t * t + 1.).sqrt();
                let s = t * c;

                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }

    let values: Vec<f32> = (0..n).map(|i| a[i][i]).collect();
    let vectors: Vec<f32> = v.into_iter().flatten().collect();
    let device = matrix.device();
    Ok((
        Tensor::from_vec(values, n, device)?,
        Tensor::from_vec(vectors, (n, n), device)?,
    ))
}
