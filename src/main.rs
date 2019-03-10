use geo::algorithm::simplifyvw::SimplifyVWPreserve;
use geo::{LineString, Point};
use piston_window::draw_state::Blend;
use piston_window::*;

fn main() {
    let mut window: PistonWindow = WindowSettings::new("piston: draw_state", [1024, 1024])
        .exit_on_esc(true)
        .samples(4)
        .build()
        .unwrap();

    let blends = [Blend::Alpha, Blend::Add, Blend::Invert, Blend::Multiply];
    let mut blend = 0;

    let mut vec = Vec::new();
    let red = [1.0, 0.0, 0.0, 1.0];
    let green = [0.0, 1.0, 0.0, 1.0];
    let mut hold_mouse = false;
    window.set_lazy(true);
    while let Some(e) = window.next() {

        let linestring = LineString::from(vec.clone());

        let simplified = linestring.simplifyvw_preserve(&1.6);

        window.draw_2d(&e, |c, mut g| {
            clear([0.8, 0.8, 0.8, 1.0], g);
            g.clear_stencil(0);
//            vec.iter().for_each(|v: &Point<f64>| {
//                draw_rect(v.x(), v.y(), red, &c, &mut g);
//            });

            simplified.points_iter().for_each(|v| {
                draw_rect(v.x(), v.y(), green, &c, &mut g);
            });

        });

        if let Some(Button::Mouse(button)) = e.press_args() {
            println!("mouse down");

            hold_mouse = true
        }

        if let Some(Button::Keyboard(Key::A)) = e.press_args() {
            println!("Changed blending to {:?}", 1);
        }

        if let Some(button) = e.release_args() {
            match button {
                Button::Mouse(button) => {
                    println!("mouse up");
                    hold_mouse = false
                },
                _ => {}
            }
        };

        e.mouse_cursor(|x, y| {
            if hold_mouse {
                println!("Mouse moved '{} {}'", x, y);
                vec.push(Point::new(x, y));

            }
        });
    }
}

fn draw_rect(x: f64, y: f64, color: [f32; 4], c: &Context, g: &mut G2d) {
    Rectangle::new(color).draw([x, y, 10.0, 10.0], &c.draw_state, c.transform, g);
}
