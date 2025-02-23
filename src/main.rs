
use std::f64::consts::PI;
use std::ops::{Index, IndexMut};
use std::usize;
use bit_vec::BitVec;
use ndarray::{Array1, Array2};
use bevy::ecs::{
    component::Component,
    system::{Query, Commands}
};
use bevy::app::{App, Startup, Update};
use bevy::math::*;
use bevy::DefaultPlugins;
use bevy::sprite::Sprite;
use bevy::prelude::*;
use rand::{thread_rng, Rng};

const COUNT_RAYS: usize = 24; 
const SCALE_SIGNIFICANCE: f64 = 0.1;
const SPEED_SIGNIFICANCE: f64 = 0.1;
const DEFAULT_VIEW_ANGLE: f64 = PI/3.; // measured angle from direction in degrees
const MIN_VIEW_ANGLE: f64 = PI/12.;
const MAX_VIEW_ANGLE: f64 = PI/1.4;
const ANGLE_CHANGE: f64 = 0.05;
const SCALE_CHANGE: f64 = 0.02;
const MAX_RAY_LENGTH: f32 = 200.;
const DEFAULT_ORGANISM_RADIUS: f64 = 10.; // organisms are circular
const MAX_SCALE: f64 = 1.2;
const MIN_SCALE: f64 = 0.7;
const PIXEL_RATIO: f32 = 1.;
const STARTING_ENERGY: i64 = 10_000;
const COUNT_ORGANISMS: usize = 700;
const ENV_PRESSURE_FACTOR: f32 = 2.;
const LEARNING_RATE: f64 = 0.2;
const MAX_START_WEIGHT: f64 = 30.;
const COUNT_INPUT: usize = 2 * COUNT_RAYS + 1; // + 1 to account for how much energy a creature has, * 2 for creature scale
const COUNT_OUTPUT: usize = 2; // rotate or forward
const COUNT_HIDDEN: usize = 8;
const ROTATION_COST: f64 = 100.;
const FORWARD_COST: f64 = 2.5;
const SPAWN_HALF_WIDTH: i64 = 900;
const SPAWN_HALF_HEIGHT: i64 = 500;
const MAX_SIMULATION_ITER_PER_GEN: usize = 1800;

type Matrix = Array2<f64>;
type Vector = Array1<f64>;

#[derive(Debug)]
struct FeedData {
    bias: Vector,
    weights: Matrix,
    activator: Option<fn(&f64) -> f64>
}

impl FeedData {
    fn neurons_from(&self, prev_neurons: &Vector) -> Vector {
        (self.weights.dot(prev_neurons) + &self.bias)
            .map(self.activator.unwrap_or(|&x| x))
    }

    fn randomized(prev_size: usize, current_size: usize, activator: Option<fn(&f64) -> f64>) -> Self {
        let mut rng = rand::thread_rng();
        let weights: Matrix = Matrix::from_shape_fn((current_size, prev_size), |_| rng.gen_range(-MAX_START_WEIGHT..MAX_START_WEIGHT));
        // start off with no bias
        let bias = Vector::zeros(current_size);
        Self {
            bias,
            weights,
            activator
        }
    }

    fn mutated(parent: &Self) -> Self {
        let mut rng = rand::thread_rng();
        
        Self{
            bias: parent.bias.map(|x| x + rng.gen_range(-LEARNING_RATE..LEARNING_RATE)),
            weights: parent.weights.map(|x| x + rng.gen_range(-LEARNING_RATE..LEARNING_RATE)),
            activator: parent.activator
        }
    }
}

#[derive(Debug)]
struct Network {
       feed_data: Vec<FeedData>
}

impl Network {
    // predicts the corresponding output for input and returns the neurons in each layer
    fn predict(&self, input: &Vector) -> Vec<Vector> {
        let mut result: Vec<Vector> = vec![];
        result.reserve(self.feed_data.len() + 1);
        result.push(input.clone());

        for i in 1..self.feed_data.len() {
            result.push(self.feed_data[i-1].neurons_from(&result[i-1]));
        }
    
        result
    }

    fn predict_final(&self, input: &Vector) -> Vector {
        let prediction = self.predict(input);
        prediction[prediction.len() - 1].clone()
    }

    fn randomized(size_layers: &[usize]) -> Self {
        Self {
            feed_data: (1..size_layers.len())
                .into_iter()
                .map(|i| (i, i - 1))
                .map(|(cur, prev)| FeedData::randomized(size_layers[prev], size_layers[cur], None))
                .collect()
        }
    }

    fn mutated(parent_net: &Self) -> Self {
        Self{
            feed_data: parent_net.feed_data
                        .iter()
                        .map(FeedData::mutated)
                        .collect()
        }
    }
}

enum OutputNeuron {
    Rotation,
    Forward
}

#[derive(Component, PartialEq, Eq, Clone, Debug)]
struct OrganismID(usize);

#[derive(Debug)]
struct Organism {
    brain: Network,
    id: OrganismID,

    color: u32,
    scale: f64,
    view_angle: f64,
   
    position: Vec2,
    rotation: f64,
    energy: i64,

    time_lived: u64,
    count_eaten: u64,
    avg_speed: f64,
}

/* Determines if a ray with x-axis angle angle from point p_x intersects the sphere with radius
 * ORGANISM RADIUS centred at point p_y
 * Returns the length/corresponding line argument if there is an intersection, None if no
 * intersection exists
 * For more information https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
*/
fn intersect_ray_sphere(p_x: Vec2, p_y: Vec2, r_y: f64, angle: f32) -> Option<f32> {
    if (p_x - p_y).length() <= r_y as f32 {
        return Some((p_x - p_y).length());
    }
    let d_x = Vec2::from_angle(angle);
    let b = 2. * d_x.dot(p_x - p_y);
    let c = (p_x - p_y).dot(p_x - p_y) - (r_y * r_y) as f32;
    let discriminant = b * b - 4. * c;
    
    if discriminant < 0. {
        return None;
    }
    
    // first root willl always be nearest intersection unless point is in circle
    let fst = (-b - discriminant.sqrt())/ 2.;

    if fst < 0. || fst > MAX_RAY_LENGTH {
        return None;
    }
    
    return Some(fst);
}

fn intersect_sphere_sphere(p_x: Vec2, p_y: Vec2, r_x: f64, r_y: f64) -> bool {
    (p_x - p_y).length() <= (r_x + r_y) as f32
}

impl Default for Organism {
    fn default() -> Self {
        let mut rng = thread_rng();
        Self {
            brain: Network::randomized(&vec![COUNT_INPUT, COUNT_HIDDEN, COUNT_OUTPUT]), 
            scale: 1.,
            view_angle: DEFAULT_VIEW_ANGLE,
            position:   vec2(
                            rng.gen_range(-SPAWN_HALF_WIDTH..SPAWN_HALF_WIDTH) as f32, 
                            rng.gen_range(-SPAWN_HALF_HEIGHT..SPAWN_HALF_HEIGHT) as f32
                        ),
            color: rng.gen_range(0x999999..0xffffff),
            id: OrganismID(0),
            rotation: rng.gen_range(-PI..PI),
            energy: STARTING_ENERGY, 
            time_lived: 0, 
            count_eaten: 0,
            avg_speed: 0.,
        }
    }
}

fn clamp(x: f64, bottom: f64, top: f64) -> f64 {
    x.max(bottom).min(top)
}

fn shift_random(x: f64, delta: f64) -> f64 {
        let mut rng = thread_rng();
    x + rng.gen_range(-delta..delta)
}

fn wraparound(position: Vec2) -> Vec2 {
    let mut res_x = position.x;
    if position.x < -SPAWN_HALF_WIDTH as f32 {
        res_x = SPAWN_HALF_WIDTH as f32;
    } else if position.x > SPAWN_HALF_WIDTH as f32 {
        res_x = -SPAWN_HALF_WIDTH as f32;
    }

    let mut res_y = position.y;
    if position.y < -SPAWN_HALF_HEIGHT as f32 {
        res_y = SPAWN_HALF_HEIGHT as f32;
    } else if position.y > SPAWN_HALF_HEIGHT as f32 {
        res_y = -SPAWN_HALF_HEIGHT as f32;
    }

    return vec2(res_x, res_y);
}

impl Organism {
    fn from(id: usize) -> Self {
        Self {
            id: OrganismID(id),
            ..Default::default()
        }
    }

    fn descendent_of(id: OrganismID, parent: &Self) -> Self {
        Self {
            brain: Network::mutated(&parent.brain),
            scale: clamp(shift_random(parent.scale, SCALE_CHANGE), MIN_SCALE, MAX_SCALE),
            view_angle: clamp(shift_random(parent.view_angle, ANGLE_CHANGE), MIN_VIEW_ANGLE, MAX_VIEW_ANGLE),
            color: parent.color,
            id,
            ..Default::default()
        }
    }

    fn get_color(&self) -> Color {
        Color::srgb_u8(((self.color & 0xff0000) >> 16) as u8, ((self.color & 0x00ff00) >> 8) as u8 , (self.color & 0x0000ff) as u8)
    }

    fn fitness(&self) -> u64 {
        self.count_eaten * self.count_eaten * self.count_eaten + self.time_lived / 100 + self.avg_speed as u64
    }

    fn perceive_single(&self, vision: &mut Vector, index: usize, angle: f64, other: &Self, is_dead: &BitVec) {
        if self.id == other.id || is_dead[other.id.0] {
            return;
        }

        let intersection = intersect_ray_sphere(
            self.position,
            other.position,
            DEFAULT_ORGANISM_RADIUS * other.scale,
            angle as f32,
        );

        if let Some(x) = intersection {
            let neural_input = 1000. / (0.2 * (x as f64) + 0.01);
            // whichever is the strongest stimulus
            vision[index] = vision[index].max(neural_input);
            if neural_input > vision[index] {
                vision[index] = neural_input;
                vision[index * COUNT_RAYS] = other.scale;
            }
        }
    }

    fn perceive_pop(&self, vision: &mut Vector, population: &[Self], is_dead: &BitVec) {
        if is_dead[self.id.0] {
            return;
        }
        
        let lerp_denom = (COUNT_RAYS - 1) as f64;
        vision.fill(0.);
        
        for k in 0..COUNT_RAYS {
            let angle = 2. * self.view_angle * ((k as f64) / lerp_denom) + self.rotation - self.view_angle;

            for organism in population.iter() {
                self.perceive_single(vision, k, angle, organism, is_dead);
            }
        }

        let n = vision.len();
        vision[n - 1] = (self.energy / 1000) as f64;
    }

    fn beats(&self, other: &Self) -> bool {
        let mut rng = thread_rng();
        let mut scale_factor = 1.0;
        let mut speed_factor = 1.0;

        if (self.scale - other.scale).abs() > SCALE_SIGNIFICANCE {
            let scale_percent_diff = (other.scale - self.scale) / self.scale;
            scale_factor = 1. - 0.1 * scale_percent_diff;
        }

        if (other.avg_speed - self.avg_speed).abs() > SPEED_SIGNIFICANCE {
            if other.avg_speed > self.avg_speed {
                speed_factor = 0.4;
            } else {
                speed_factor = 1.6;
            }
        }
        
        rng.gen_range(0.0..1.0) <= 0.5 * scale_factor * speed_factor
    }

    fn update(&mut self, vision: &Vector, is_dead: &BitVec, time: &Res<Time>, transform: &mut Transform) {
        if is_dead[self.id.0] {
            return;
        }

        let elapsed = time.delta_secs() as f64;
        let decision = self.brain.predict_final(vision);
        let forward = 0.02 * (-2. * self.scale + 2.).exp() * decision[OutputNeuron::Forward as usize].abs() * elapsed;
        let rotation_delta = 0.015 * decision[OutputNeuron::Rotation as usize] * elapsed;
        self.energy -= (forward * FORWARD_COST * self.scale) as i64;
        self.energy -= (rotation_delta * ROTATION_COST * self.scale) as i64;

        self.rotation += rotation_delta;
        let velocity = (forward as f32) * Vec2::from_angle(self.rotation as f32);
        let velocity_len = forward;
        self.position = wraparound(self.position + velocity);
        transform.translation = vec3(self.position.x, self.position.y, 1.);
        transform.rotate_z(rotation_delta as f32);
        self.avg_speed = self.avg_speed * 0.6 + (velocity_len as f64) * 0.4;
    }
}

struct Population {
    organisms: Vec<Organism>,
    sense_data: Vec<Vector>,
    is_dead: BitVec,
}

impl Index<&OrganismID> for Population {
    type Output = Organism;
    fn index(&self, index: &OrganismID) -> &Self::Output {
        let idx = index.0;  // Get the usize from the X struct
        &self.organisms[idx]
    }
}

impl IndexMut<&OrganismID> for Population {
    fn index_mut(&mut self, index: &OrganismID) -> &mut Organism {
        let idx = index.0;  // Get the usize from the X struct
        &mut self.organisms[idx]
    }
}

impl Population {
    fn seed_random() -> Self {
        Population { 
            organisms: (0..COUNT_ORGANISMS)
                .into_iter()
                .map(|i| Organism::from(i))
                .collect(),
            sense_data: (0..COUNT_ORGANISMS)
                .into_iter()
                .map(|_| Vector::zeros(COUNT_INPUT))
                .collect(),
            is_dead: BitVec::from_elem(COUNT_ORGANISMS, false),
        }
    }

    fn next_generation(&mut self) {
        let previous_population = &mut self.organisms;
        
        previous_population.sort_by(|x, y| y.fitness().cmp(&x.fitness()));  

        let mut rng = rand::thread_rng();
        let n = (COUNT_ORGANISMS / (ENV_PRESSURE_FACTOR as usize)) as usize;
        let organisms = (0..COUNT_ORGANISMS).into_iter()
                        .map(|i| (i, rng.gen_range(0..n)))
                        .map(|(i, rand_idx)| Organism::descendent_of(OrganismID(i), &previous_population[rand_idx]))
                        .collect();
        self.organisms = organisms;
        self.is_dead.clear();
    }
    
    fn perceive(&mut self) {
        for organism in self.organisms.iter() {
            organism.perceive_pop(&mut self.sense_data[organism.id.0], &self.organisms, &self.is_dead);
        }
    }

    fn update_organism(&mut self, id: &OrganismID, time: &Res<Time>, transform: &mut Transform) -> bool {
        if self.is_dead[id.0] {
            return false;
        }

        self.organisms[id.0].update(&self.sense_data[id.0], &self.is_dead, time, transform);
        return true;
    }
}

#[derive(Resource)]
struct Simulation {
    population: Population,
    generation_num: usize,
    count_step: usize,
    creature_image: Handle<Image>
}

impl Simulation {
    fn next_generation(&mut self, commands: &mut Commands, query: &mut Query<(&OrganismID, &mut Transform, Entity)>) {
        self.clear_entities(commands, query);
        self.generation_num += 1;
        println!("Generation {}", self.generation_num);
        self.count_step = 0;
        self.population.next_generation();
        self.spawn(commands);
    }

    fn clear_entities(&self, commands: &mut Commands, query: &mut Query<(&OrganismID, &mut Transform, Entity)>) {
        for (_, _, entity) in query.iter_mut() {
            commands.entity(entity).despawn();
        }
    }

    fn spawn(&self, commands: &mut Commands) {
        for (i, organism) in self.population.organisms.iter().enumerate() {
            commands.spawn((
                    Sprite {
                        image: self.creature_image.clone(),
                        color: organism.get_color(),
                        ..Default::default()
                    },
                    Transform{
                        translation: vec3(organism.position.x, organism.position.y, 1.),
                        rotation: Quat::from_rotation_z((organism.rotation - PI/2.) as f32),
                        ..Default::default()
                    }
                    .with_scale(Vec3::splat(PIXEL_RATIO * (organism.scale as f32))),
                    OrganismID(i)
            ));
        }
    }

    fn clear_dead(&self, commands: &mut Commands, query: &mut Query<(&OrganismID, &mut Transform, Entity)>) {
        for (id, _, entity) in query.iter_mut() {
            if self.population.is_dead[id.0] {
                commands.entity(entity).despawn();
            }
        }
    }

    fn update_organism(&mut self, id: &OrganismID, time: &Res<Time>, transform: &mut Transform) -> bool {
        return self.population.update_organism(id, time, transform);
    }
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let organism_image_handle = asset_server.load("creature2.png");

    let simulation = Simulation{
        population: Population::seed_random(),
        count_step: 0,
        generation_num: 0,
        creature_image: organism_image_handle.clone(),
    };

    println!("Generation {}", 0);
    commands.insert_resource(ClearColor(Color::srgb(0.8, 0.8, 0.8)));
    commands.spawn(Camera2d::default());

    simulation.spawn(&mut commands);
    commands.insert_resource(simulation);
}

fn prepare_organisms(mut simulation: ResMut<Simulation>) {
    let population = &mut simulation.population;
    population.perceive();
}

fn update_organisms(
    mut commands: Commands, 
    mut simulation: ResMut<Simulation>, 
    time: Res<Time>, 
    mut query: Query<(&OrganismID, &mut Transform, Entity)>
) {
    let mut is_dead = simulation.population.is_dead.clone();

    for (id, mut transform, _) in query.iter_mut() {
        if is_dead[id.0] {
            continue;
        }

        simulation.update_organism(id, &time, &mut transform);
    }
    
    let n = simulation.population.organisms.len();
    for i in 0..n {
        if is_dead[i] {
            continue;
        }
        for j in i+1..n {
            if is_dead[j] {
                continue;
            }

            if !intersect_sphere_sphere(
                    simulation.population.organisms[i].position, 
                    simulation.population.organisms[j].position,
                    DEFAULT_ORGANISM_RADIUS * simulation.population.organisms[i].scale,
                    DEFAULT_ORGANISM_RADIUS * simulation.population.organisms[j].scale,
            ) {
                continue;
            }

            if simulation.population.organisms[i].beats(&simulation.population.organisms[j]) {
                is_dead.set(j, true);
                simulation.population.organisms[i].count_eaten += 1;
            } else {
                is_dead.set(i, true);
                simulation.population.organisms[j].count_eaten += 1;
            }
        }
    }

    let mut any_alive = false;
    for organism in simulation.population.organisms.iter() {
        if organism.energy <= 0 {
            is_dead.set(organism.id.0, true);
        } else {
            any_alive = true;
        }
    }

    for organism in simulation.population.organisms.iter_mut() {
        if !is_dead[organism.id.0] {
            organism.time_lived += 1;
        }
    }
    
    simulation.population.is_dead = is_dead;
    simulation.count_step += 1;
    if simulation.count_step > MAX_SIMULATION_ITER_PER_GEN || !any_alive {
        simulation.next_generation(&mut commands, &mut query);
        return;
    }
    
    simulation.clear_dead(&mut commands, &mut query);
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, (prepare_organisms, update_organisms).chain())
        .run();
}
