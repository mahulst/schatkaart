use imageproc::drawing::draw_cubic_bezier_curve_mut;
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::rect::Rect;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use other_image::Rgba;
use other_image::RgbaImage;
use piston_window::draw_state::Blend;
use piston_window::*;
use std::cmp::max;
use std::ops::Add;
use std::path::Path;

#[derive(PartialEq, Debug, Copy, Clone)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Add for Vec2 {
    type Output = Vec2;

    fn add(self, other: Vec2) -> Vec2 {
        Vec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

const red: [u8; 4] = [255, 0, 0, 255];
const green: [u8; 4] = [0, 255, 0, 255];
const blue: [u8; 4] = [0, 0, 255, 255];
const white: [u8; 4] = [255, 255, 255, 255];

fn draw_curve(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, image: &mut RgbaImage) {
    draw_cubic_bezier_curve_mut(
        image,
        (p0.x, p0.y),
        (p3.x, p3.y),
        (p1.x, p1.y),
        (p2.x, p2.y),
        Rgba(blue),
    );
    for point in [p0, p1, p2, p3].iter() {
        let rect = Rect::at(point.x as i32, point.y as i32).of_size(5, 5);
        debug!("point: {:?}", point);
        draw_filled_rect_mut(image, rect, Rgba(red));
    }
}

fn get_image(vec: &Vec<Vec2>) -> RgbaImage {
    let mut image = RgbaImage::new(1024, 1024);

    // Draw a line segment wholly within bounds
    draw_curve(vec[0], vec[1], vec[2], vec[3], &mut image);

    let path = Path::new("./tmp.png");
    image.save(path).unwrap();
    image
}

fn main() {
    simple_logger::init().unwrap();
    let mut window: PistonWindow = WindowSettings::new("piston: draw_state", [1024, 1024])
        .exit_on_esc(true)
        .samples(4)
        .build()
        .unwrap();

    let assets = find_folder::Search::ParentsThenKids(3, 3)
        .for_folder("assets")
        .unwrap();
    let blends = [Blend::Alpha, Blend::Add, Blend::Invert, Blend::Multiply];
    let mut blend = 0;
    let mut clip_inside = true;
    let mut hold_mouse = false;
    let mut vec: Vec<Vec2> = vec![
        Vec2 { x: 100.0, y: 250.0 },
        Vec2 { x: 200.0, y: 700.0 },
        Vec2 { x: 700.0, y: 700.0 },
        Vec2 { x: 800.0, y: 250.0 },
    ];
    let mut cursor = Vec2 { x: 0.0, y: 0.0 };
    window.set_lazy(true);

    let mut img_data = get_image(&vec);
    let mut img =
        Texture::from_image(&mut window.factory, &img_data, &TextureSettings::new()).unwrap();

    while let Some(e) = window.next() {
        window.draw_2d(&e, |c, g| {
            clear([0.8, 0.8, 0.8, 1.0], g);
            g.clear_stencil(0);

            let transform = c.transform.trans(0.0, 0.0);
            Image::new().draw(&img, &DrawState::new_outside(), transform, g);
        });

        if let Some(Button::Mouse(button)) = e.press_args() {
            println!("mouse down");
            vec.push(cursor.clone());

            vec.iter().for_each(|v| {
                info!("{:?}", v);
            });

            hold_mouse = true
        }

        if let Some(Button::Keyboard(Key::A)) = e.press_args() {
            println!("Changed blending to {:?}", 1);
            img_data = get_image(&vec);
            img = Texture::from_image(&mut window.factory, &img_data, &TextureSettings::new())
                .unwrap();
        }

        if let Some(button) = e.release_args() {
            match button {
                Button::Mouse(button) => {
                    println!("mouse up");
                    hold_mouse = false
                }
                _ => {}
            }
        };

        e.mouse_cursor(|x, y| {
            cursor = Vec2 {
                x: x as f32,
                y: y as f32,
            };
        });
    }
}

fn draw_rect(x: f64, y: f64, color: [f32; 4], c: &Context, g: &mut G2d) {
    Rectangle::new(color).draw([x, y, 10.0, 10.0], &c.draw_state, c.transform, g);
}

#[derive(PartialEq, Debug)]
struct CubicBezier {
    s: Vec2,
    e: Vec2,
    c1: Vec2,
    c2: Vec2,
}

impl CubicBezier {
    pub fn q(&self, t: f32) -> Vec2 {
        debug!("s {:?}", self.s);
        debug!("c1 {:?}", self.c1);
        debug!("c2 {:?}", self.c2);
        debug!("e {:?}", self.e);
        debug!("t {}", t);
        let tx = 1.0 - t;
        let pA = Math::mul_items(&self.s, tx * tx * tx);
        let pB = Math::mul_items(&self.c1, 3.0 * tx * tx * t);
        let pC = Math::mul_items(&self.c2, 3.0 * tx * t * t);
        let pD = Math::mul_items(&self.e, t * t * t);
        debug!("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
        debug!("pA: {:?}", pA);
        debug!("pB: {:?}", pB);
        debug!("pC: {:?}", pC);
        debug!("pD: {:?}", pD);
        debug!("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
        pA + pB + pC + pD
    }

    pub fn qprime(&self, t: f32) -> Vec2 {
        let tx = 1.0 - t;
        let pA = Math::mul_items(&Math::subtract(&self.c1, &self.s), 3.0 * tx * tx);
        let pB = Math::mul_items(&Math::subtract(&self.c2, &self.c1), 6.0 * tx * t);
        let pC = Math::mul_items(&Math::subtract(&self.e, &self.c2), 3.0 * t * t);

        pA + pB + pC
    }
    pub fn qprimeprime(&self, t: f32) -> Vec2 {
        let a = Math::mul_items(&self.c1, 2.0);
        let a = Math::subtract(&self.c2, &a);
        let a = Vec2 {
            x: self.s.x + a.x,
            y: self.s.y + a.y,
        };
        let a = Math::mul_items(&a, 6.0 * (1.0 - t));

        let b = Math::mul_items(&self.c2, 2.0);
        let b = Math::subtract(&self.e, &b);
        let b = self.c1 + b;
        let b = Math::mul_items(&b, 6.0 * t);

        a + b
    }
}

fn fit_curve(points: &Vec<Vec2>, max_error: f32) -> Vec<CubicBezier> {
    let len = points.len();
    if len < 2 {
        vec![]
    } else {
        let left_tangent = create_tangent(&points[1], &points[0]);
        let right_tangent = create_tangent(&points[len - 2], &points[len - 1]);
        //        debug!("{:?}", left_tangent);
        //        debug!("{:?}", right_tangent);
        fit_cubic(&points, &left_tangent, &right_tangent, max_error)
    }
}

fn create_tangent(p1: &Vec2, p2: &Vec2) -> Vec2 {
    Math::normalize(&Math::subtract(p1, p2))
}

fn fit_cubic(
    points: &[Vec2],
    left_tangent: &Vec2,
    right_tangent: &Vec2,
    error: f32,
) -> Vec<CubicBezier> {
    const MAX_ITERATIONS: u8 = 20;

    if points.len() == 2 {
        let dist = Math::vector_len(&Math::subtract(&points[0], &points[1]));
        vec![CubicBezier {
            s: points[0],
            c1: Math::add(&points[0], &Math::multiply(left_tangent, dist)),
            c2: Math::add(&points[1], &Math::multiply(right_tangent, dist)),
            e: points[1],
        }]
    } else {
        let params = chord_length_parameterize(points);
        let (bez_curve, max_error, split_point) =
            generate_and_report(points, &params, &left_tangent, &right_tangent);

        if max_error < error {
            return vec![bez_curve];
        }

        if max_error < error * error {
            let mut prev_error = max_error;
            let mut prev_split = split_point;

            for i in 0..MAX_ITERATIONS {
                let u_prime = reparameterize(&bez_curve, points, &params);

                let (bez_curve, max_error, split_point) =
                    generate_and_report(&points, &params, &left_tangent, &right_tangent);

                if max_error < error {
                    return vec![bez_curve];
                } else if split_point == prev_split {
                    let err_change = max_error / prev_error;

                    if err_change > 0.9999 && err_change < 1.0001 {
                        break;
                    }
                }
                prev_error = max_error;
                prev_split = split_point;
            }
        }

        let mut beziers = vec![];

        let mut center_vector = Math::subtract(
            &points[split_point as usize - 1],
            &points[split_point as usize + 1],
        );

        if center_vector.x == 0.0 && center_vector.y == 0.0 {
            center_vector = Math::subtract(
                &points[split_point as usize - 1],
                &points[split_point as usize],
            );
            center_vector.x = -center_vector.y;
            center_vector.y = center_vector.x
        }

        let to_center_tangent = Math::normalize(&center_vector);

        let from_center_tangent = Math::mul_items(&to_center_tangent, -1.0);

        beziers = fit_cubic(
            &points[0..split_point as usize + 1],
            left_tangent,
            &to_center_tangent,
            error,
        );
        let beziers2 = fit_cubic(
            &points[(split_point as usize)..],
            &from_center_tangent,
            right_tangent,
            error,
        );

        beziers.extend(beziers2);

        beziers
    }
}

fn reparameterize(curve: &CubicBezier, points: &[Vec2], params: &Vec<f32>) -> Vec<f32> {
    let mut params_prime = vec![];

    for i in 0..params.len() {
        params_prime.push(newton_raphson_root_find(curve, &points[i], params[i]))
    }

    params_prime
}

fn newton_raphson_root_find(curve: &CubicBezier, point: &Vec2, u: f32) -> f32 {
    let d = Math::subtract(&curve.q(u), point);
    let q_prime = curve.qprime(u);
    let numerator = Math::dot(&d, &q_prime);
    let squared = q_prime.x * q_prime.x + q_prime.y * q_prime.y;
    let qprimeprime = curve.qprimeprime(u);
    let multiplied = d.x * qprimeprime.x + d.y * qprimeprime.y;
    let denominator = squared + 2.0 * multiplied;

    if denominator == 0.0 {
        u
    } else {
        u - (numerator / denominator)
    }
}

fn generate_and_report(
    points: &[Vec2],
    params: &Vec<f32>,
    left_tangent: &Vec2,
    right_tangent: &Vec2,
) -> (CubicBezier, f32, f32) {
    let bez_curve = generate_bezier(points, params, left_tangent, right_tangent);

    let (max_error, split_point) = compute_max_error(points, &bez_curve, params);

    (bez_curve, max_error, split_point)
}

fn generate_bezier(
    points: &[Vec2],
    params: &Vec<f32>,
    left_tangent: &Vec2,
    right_tangent: &Vec2,
) -> CubicBezier {
    let mut curve = CubicBezier {
        s: points[0],
        e: points[points.len() - 1],
        c1: Vec2 { x: 0.0, y: 0.0 },
        c2: Vec2 { x: 0.0, y: 0.0 },
    };
    let mut A = Math::zeros_xx2x2(params.len() as u32);

    for i in 0..params.len() {
        let u = params[i];
        let ux = 1.0 - u;

        let a = (
            Math::multiply(&left_tangent, 3.0 * u * (ux * ux)),
            Math::multiply(&right_tangent, 3.0 * ux * (u * u)),
        );
        A[i] = a;
    }

    let mut C = (Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 0.0, y: 0.0 });
    let mut X = Vec2 { x: 0.0, y: 0.0 };

    for i in 0..params.len() {
        let u = params[i];
        let a = A[i];

        C.0.x += Math::dot(&a.0, &a.0);
        C.0.y += Math::dot(&a.0, &a.1);
        C.1.x += Math::dot(&a.0, &a.1);
        C.1.y += Math::dot(&a.1, &a.1);
        debug!("q(): {:?}", curve.q(u));

        let tmp = Math::subtract(
            &points[i],
            &CubicBezier {
                s: curve.s,
                c1: curve.s,
                c2: curve.e,
                e: curve.e,
            }
            .q(u),
        );

        X.x += Math::dot(&a.0, &tmp);
        X.y += Math::dot(&a.1, &tmp);
    }

    debug!("C: {:?}", C);
    debug!("X: {:?}", X);
    let det_C0_C1 = (C.0.x * C.1.y) - (C.1.x * C.0.y);
    let det_C0_X = (C.0.x * X.y) - (C.1.x * X.y);
    let det_X_C1 = (X.x * C.1.y) - (X.y * C.0.y);

    let alpha_l = if det_C0_C1 == 0.0 {
        0.0
    } else {
        det_X_C1 / det_C0_C1
    };
    let alpha_r = if det_C0_C1 == 0.0 {
        0.0
    } else {
        det_C0_X / det_C0_C1
    };
    let first_point = points.first().unwrap();
    let last_point = points.last().unwrap();
    let seg_length = Math::vector_len(&Math::subtract(first_point, last_point));
    let epsilon = 1.0e-6 * seg_length;

    if alpha_l < epsilon || alpha_r < epsilon {
        curve.c1 = *first_point + Math::mul_items(left_tangent, seg_length / 3.0);
        curve.c2 = *last_point + Math::mul_items(right_tangent, seg_length / 3.0);
    } else {
        curve.c1 = *first_point + Math::mul_items(left_tangent, alpha_l);
        curve.c2 = *last_point + Math::mul_items(right_tangent, alpha_r);
    }

    curve
}

fn compute_max_error(points: &[Vec2], bez_curve: &CubicBezier, params: &Vec<f32>) -> (f32, f32) {
    let mut max_dist = 0.0;
    let mut split_point = (points.len() as f32) / 2.0;

    let t_dist_map = map_t_to_relative_distances(bez_curve, 10);

    for i in 0..points.len() {
        let point = points[i];
        let t = find_t(bez_curve, params[i], &t_dist_map, 10);
        let v = Math::subtract(&bez_curve.q(t), &point);

        let dist = v.x * v.x + v.y * v.y;

        if dist < max_dist {
            max_dist = dist;
            split_point = i as f32;
        }
    }

    (max_dist, split_point)
}

fn map_t_to_relative_distances(curve: &CubicBezier, parts: usize) -> Vec<f32> {
    let mut sum_len = 0.0;
    let mut b_t_prev = curve.s;
    let mut b_t_dist = vec![0.0];
    for i in 1..parts {
        let b_t_curr = curve.q((i as f32) / (parts as f32));

        sum_len += Math::vector_len(&Math::subtract(&b_t_curr, &b_t_prev));
        b_t_dist.push(sum_len.clone());
        b_t_prev = b_t_curr;
    }

    b_t_dist.iter().map(|x| x / sum_len).collect()
}

fn find_t(curve: &CubicBezier, param: f32, t_dist_map: &Vec<f32>, parts: usize) -> f32 {
    if param < 0.0 {
        return 0.0;
    };
    if param > 1.0 {
        return 1.0;
    };

    let mut t = 0.0;
    for i in 1..parts {
        if param <= t_dist_map[i] {
            let t_min = (i - 1) / parts;
            let t_max = i / parts;
            let len_min = t_dist_map[i - 1];
            let len_max = t_dist_map[i];

            t = (param - len_min) / (len_max - len_min) * ((t_max as f32) - (t_min as f32))
                + (t_min as f32);
            break;
        }
    }
    t
}

fn chord_length_parameterize(points: &[Vec2]) -> Vec<f32> {
    let mut u = vec![];
    let mut i = 0;
    let mut currU = 0.0;
    let mut prevU = 0.0;
    let mut prevP = Vec2 { x: 0.0, y: 0.0 };
    points.iter().for_each(|p| {
        currU = if i > 0 {
            prevU + Math::vector_len(&Math::subtract(p, &prevP))
        } else {
            0.0
        };
        u.push(currU);

        prevU = currU;
        prevP = p.clone();
        i += 1;
    });

    return u.iter().map(|v| v / prevU).collect();
}

struct Math;

impl Math {
    pub fn zeros_xx2x2(x: u32) -> Vec<(Vec2, Vec2)> {
        let mut zs = vec![];
        for i in 0..x {
            zs.push((Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 0.0, y: 0.0 }));
        }
        zs
    }
    pub fn multiply(v: &Vec2, mul: f32) -> Vec2 {
        Vec2 {
            x: v.x * mul,
            y: v.y * mul,
        }
    }
    pub fn dot(v1: &Vec2, v2: &Vec2) -> f32 {
        v1.x * v2.x + v1.y * v2.y
    }

    pub fn add(v1: &Vec2, v2: &Vec2) -> Vec2 {
        Vec2 {
            x: v1.x + v2.x,
            y: v1.y + v2.y,
        }
    }

    pub fn subtract(v1: &Vec2, v2: &Vec2) -> Vec2 {
        Vec2 {
            x: v1.x - v2.x,
            y: v1.y - v2.y,
        }
    }
    pub fn vector_len(v: &Vec2) -> f32 {
        (v.x).hypot(v.y)
    }
    pub fn mul_items(v: &Vec2, multiplier: f32) -> Vec2 {
        Vec2 {
            x: v.x * multiplier,
            y: v.y * multiplier,
        }
    }
    pub fn normalize(v: &Vec2) -> Vec2 {
        let vector_len = Self::vector_len(v);
        Vec2 {
            x: v.x / vector_len,
            y: v.y / vector_len,
        }
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.

    use crate::{fit_curve, CubicBezier, Vec2};

    #[test]
    fn test_match_example() {
        simple_logger::init().unwrap();

        let expectedResult = CubicBezier {
            s: Vec2 { x: 0.0, y: 0.0 },
            c1: Vec2 {
                x: 20.27317402,
                y: 20.27317402,
            },
            c2: Vec2 {
                x: -1.24665147,
                y: 0.0,
            },
            e: Vec2 { x: 20.0, y: 0.0 },
        };
        let points = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 10.0, y: 10.0 },
            Vec2 { x: 10.0, y: 0.0 },
            Vec2 { x: 20.0, y: 0.0 },
        ];
        let result = fit_curve(&points, 50.0);

        assert_eq!(vec![expectedResult], result);
    }

}
