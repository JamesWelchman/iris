// Iris dataset NN
use rand::prelude::SliceRandom;
use rand::distributions::Distribution;
use statrs::distribution::{Normal, Uniform};

const FILENAME: &'static str = "iris.csv";
const TEST_DATA: &'static str = "iris_test.csv";
const NORM_CONST: f64 = 10.0;

const NUM_NODES: usize = 8;
const MIDDLE_LAYER_TRAIN_RATIO: f64 = 1000.0;
const LEARNING_RATE: f64 = 0.01;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Copy, Clone, Debug)]
enum Species {
	Unknown,
	Setosa,
	Versicolor,
	Virginica,
}

impl Species {
	fn from_string(s: &str) -> Result<Self> {
		match s {
			"Iris-setosa" => Ok(Self::Setosa),
			"Iris-versicolor" => Ok(Self::Versicolor),
			"Iris-virginica" => Ok(Self::Virginica),
			_ => panic!("unrecognised species")
		}
	}
}

impl Default for Species {
	fn default() -> Self {
		Self::Unknown
	}
}

#[derive(Default, Copy, Clone, Debug)]
struct Sample {
	id: u8,
	sepal_length: f64,
	sepal_width: f64,
	petal_length: f64,
	petal_width: f64,
	species: Species,
}

fn read_samples(name: &str) -> Result<Vec<Sample>> {
	use std::io::{BufReader, BufRead};
	use std::fs::File;

	let reader = BufReader::new(File::open(name)?);

	// Read the header
	let mut lines = reader.lines();

	lines.next();

	let mut samples = vec![];

	for l in lines {
		let l = l?;
		let mut s = Sample::default();
		for (n, f) in l.split(",").enumerate() {
			match n {
				0 => s.id = f.parse::<u8>()?,
				1 => s.sepal_length = f.parse::<f64>()? / NORM_CONST,
				2 => s.sepal_width = f.parse::<f64>()? / NORM_CONST,
				3 => s.petal_length = f.parse::<f64>()? / NORM_CONST,
				4 => s.petal_width = f.parse::<f64>()? / NORM_CONST,
				5 => s.species = Species::from_string(f)?,
				_ => panic!("bad csv data")
			}
		}
		samples.push(s);
	}

	Ok(samples)
}

#[derive(Debug)]
struct Weights {
	input_layer: [[f64; 4]; NUM_NODES],

	// Our middle layer. This is dense from the layer before.
	// We apply a leaky relu afterwards.
	middle_layer: [[f64; NUM_NODES]; NUM_NODES],

	// Our output layer
	output_layer: [[f64; NUM_NODES]; 3],
}

struct NeuralNet {
	weights: Box<Weights>,

	// Our buffers
	input: [f64; 4],
	input_comp: Box<[f64; NUM_NODES]>,
	input_comp_relu: Box<[f64; NUM_NODES]>,
	middle_comp: Box<[f64; NUM_NODES]>,
	middle_comp_relu: Box<[f64; NUM_NODES]>,
	output_comp: [f64; 3],
	output_comp_softmax: [f64; 3],
}

impl NeuralNet {
	fn new() -> Result<Self> {
		let mut weights = Box::new(
			Weights{
				input_layer: [[0.0; 4]; NUM_NODES],
				middle_layer: [[0.0; NUM_NODES]; NUM_NODES],
				output_layer: [[0.0; NUM_NODES]; 3],
			}
		);
		let mut r = rand::thread_rng();
		let t_uniform = Uniform::new(-1.0, 1.0)?;
		let mut t_normal = TruncatedNormal::new(0.0, 0.5)?;

		weights.output_layer.iter_mut().flatten()
			.for_each(|w| *w = t_normal.sample());

		weights.middle_layer.iter_mut().flatten()
			.chain(weights.input_layer.iter_mut().flatten())
			.for_each(|w| *w = t_uniform.sample(&mut r));

		Ok(Self{
			weights: weights,
			input: [0.0; 4],
			input_comp: Box::new([0.0; NUM_NODES]),
			input_comp_relu: Box::new([0.0; NUM_NODES]),
			middle_comp: Box::new([0.0; NUM_NODES]),
			middle_comp_relu: Box::new([0.0; NUM_NODES]),
			output_comp: [0.0; 3],
			output_comp_softmax: [0.0; 3],
		})
	}

	fn compute(&mut self, input: [f64; 4]) -> [f64; 3] {
		self.input = input;

		for ((node, o), s) in self.weights.input_layer.iter()
			.zip(self.input_comp.iter_mut())
			.zip(self.input_comp_relu.iter_mut()) {

			*o = node.iter()
				.zip(input.iter())
				.map(|(&w, &i)| w * i)
				.sum::<f64>();

			// Apply the ReLU
			*s = sigmoid(*o);
		}

		self.compute_middle()
	}

	fn train_input(&mut self,
		           learning_rate: f64,
		           d_l_d_out: &[f64; NUM_NODES]) {

		// Step back through the sigmoid
		let d_l_d_t = d_l_d_out.iter()
			.zip(self.input_comp.iter().map(|&i| sigmoid(i)))
			.map(|(&t, i)| t * i * (1.0 - i))
			.collect::<Vec<f64>>();

		// Update the weights
		for (node, &t) in self.weights.input_layer.iter_mut()
			.zip(d_l_d_t.iter()) {

			for (w, &i) in node.iter_mut()
				.zip(self.input.iter()) {

				*w -= learning_rate * t * i;
			}
		}
	}

	fn compute_middle(&mut self) -> [f64; 3] {
		for ((node, v), r) in self.weights.middle_layer.iter()
			.zip(self.middle_comp.iter_mut())
			.zip(self.middle_comp_relu.iter_mut()) {

			*v = node.iter()
				.zip(self.input_comp_relu.iter())
				.map(|(&weight, &input)| weight * input)
				.sum::<f64>();

			// Apply the relu
			*r = if *v > 0.0 {
				*v
			} else {
				0.01 * *v
			};
		}

		self.compute_out()
	}

	fn train_middle(&mut self,
		            learning_rate: f64,
		            d_l_d_out: Box<[f64; NUM_NODES]>) {

		// Step back through leaky relu
		let mut d_l_d_t = d_l_d_out.clone();
		for (i, &v) in d_l_d_t.iter_mut()
			.zip(self.middle_comp.iter()) {
			if v <= 0.0 {
				*i *= 0.01;
			}
		}

		// Update the weights
		for (node, &t) in self.weights.middle_layer.iter_mut()
			.zip(d_l_d_t.iter()) {

			// We need to evaluate each node at the input
			// given from the previous layer
			for (w, &i) in node.iter_mut()
				.zip(self.input_comp_relu.iter()) {

				*w -= learning_rate * t * i;
			}
		}


		// Compute d_l_d_i
		let mut d_l_d_i = Box::new([0.0 as f64; NUM_NODES]);
		for ((n, v), &i) in d_l_d_i.iter_mut().enumerate()
			.zip(self.input_comp_relu.iter()) {

			*v = self.weights.middle_layer.iter()
				.zip(d_l_d_t.iter())
				.map(|(&w, &t)| w[n] * t)
				.sum::<f64>()
				* i;
		}
		// TODO
		self.train_input(learning_rate * MIDDLE_LAYER_TRAIN_RATIO * 0.1,
			             &d_l_d_i);
	}

	fn compute_out(&mut self) -> [f64; 3] {
		for (node, out) in self.weights.output_layer.iter()
			.zip(self.output_comp.iter_mut()) {

			*out = node.iter()
				.zip(self.middle_comp_relu.iter())
				.map(|(&weight, &input)| weight * input)
				.sum::<f64>();
		}

		// Compute softmax
		self.output_comp_softmax = softmax(self.output_comp);
		self.output_comp_softmax
	}

	fn train_out(&mut self,
		         learning_rate: f64,
		         d_loss_d_out: f64,
		         ans: usize) {

		// Step back through the softmax
		let mut d_l_d_t = [d_loss_d_out; 3];
		let sum_exp = self.output_comp.iter()
			.map(|v| v.exp())
			.sum::<f64>();
		let sum_exp_2 = sum_exp.powi(2);
		let class_exp = self.output_comp[ans].exp();

		for (n, t) in d_l_d_t.iter_mut().enumerate() {
			if n == ans {
				*t *= (class_exp * (sum_exp - class_exp))
					/ sum_exp_2;
			} else {
				let exp = self.output_comp[n].exp();
				*t *= (-1.0 * exp * class_exp) / sum_exp_2;
			}
		}

		// Update the weights
		for (node, &t) in self.weights.output_layer.iter_mut()
			.zip(d_l_d_t.iter()) {

			for (w, &i) in node.iter_mut()
				.zip(self.middle_comp_relu.iter()) {

				*w -= learning_rate * t * i;
			}
		}

		// Compute d_l_d_i
		let mut d_l_d_i = Box::new([0.0 as f64; NUM_NODES]);
		for ((ind, i), &v) in d_l_d_i.iter_mut().enumerate()
			.zip(self.middle_comp_relu.iter()) {

			*i = self.weights.output_layer.iter()
				.zip(d_l_d_t.iter())
				.map(|(node, &t)| node[ind] * t)
				.sum::<f64>()
				* v;
		}

		self.train_middle(learning_rate / MIDDLE_LAYER_TRAIN_RATIO,
			              d_l_d_i);
	}

	fn train(&mut self,
		     learning_rate: f64,
		     ans: usize) {

		let d_loss_d_out = -1.0 / self.output_comp_softmax[ans];
		self.train_out(learning_rate, d_loss_d_out, ans)
	}
}

fn train(nn: &mut NeuralNet) -> Result<()> {
	let mut samples = read_samples(FILENAME)?;
	let tests = read_samples(TEST_DATA)?;
	let mut rng = rand::thread_rng();

	// Read / Create NN
 	for j in 0.. {
		if j % 10_000 == 0 {
			println!("done {}", j);
			let total_loss = test(nn, &tests)?;
			if total_loss < 0.01 {

				println!("net trained - breaking\n");
				// Print tests against our training data
				test(nn, &samples)?;

				// Dump the final weights
				println!("Weights=======");
				println!("==============");
				println!("{:?}", nn.weights);
				break;
			}
		}

		samples.shuffle(&mut rng);
		for &s in samples.iter() {
			let ans = nn.compute([s.sepal_length,
				        s.sepal_width,
				        s.petal_width,
				        s.petal_length]);
			if ans[0].is_nan() || ans[1].is_nan() || ans[2].is_nan() {
				break;
			}

			nn.train(LEARNING_RATE, match s.species{
				Species::Setosa => 0,
				Species::Versicolor => 1,
				Species:: Virginica => 2,
				_ => panic!("unknown species"),
			});
		}
	}

	Ok(())
}

fn test(nn: &mut NeuralNet, samples: &[Sample]) -> Result<f64> {
	let mut total_loss = 0.0;

	for &s in samples.iter() {
		print!("[{}]\t", s.id);
		print!("species = {:?}\t", s.species);
		let ans = nn.compute([s.sepal_length,
		   		              s.sepal_width,
				              s.petal_width,
					          s.petal_length]);

		print!("[{:.4}, {:.4}, {:.4}]\t", ans[0], ans[1], ans[2]);
		let loss = -1.0 * match s.species {
			Species::Setosa => ans[0].ln(),
			Species::Versicolor => ans[1].ln(),
			Species::Virginica => ans[2].ln(),
			_ => panic!("unknown species"),

		};
		println!("loss = {:.4}", loss);
		total_loss += loss;
	}
	println!("\ntotal loss = {:.4}", total_loss);

	Ok(total_loss)
}

struct TruncatedNormal {
	norm: statrs::distribution::Normal,
	rng: rand::rngs::ThreadRng,
	min: f64,
	max: f64,
}

impl TruncatedNormal {
	fn new(mean: f64, sd: f64) -> Result<Self> {
		let r = rand::thread_rng();
		let n = Normal::new(mean, sd)?;
		Ok(TruncatedNormal{
			norm: n,
			rng: r,
			min: mean - 2.0 * sd,
			max: mean + 2.0 * sd,
		})
	}

	fn sample(&mut self) -> f64 {
		loop {
			let s = self.norm.sample(&mut self.rng) as f64;
			if s < self.max && s > self.min {
				return s;
			}
		}
	}
}

fn softmax(vals: [f64; 3]) -> [f64; 3] {
	let mut exps = [vals[0].exp(),
		        	vals[1].exp(),
			    	vals[2].exp()];
	let sum = exps.iter().sum::<f64>();
	exps[0] /= sum;
	exps[1] /= sum;
	exps[2] /= sum;

	[exps[0], exps[1], exps[2]]
}

fn sigmoid(val: f64) -> f64 {
	1.0 / (1.0 + ((-1.0 * val).exp()))
}

fn run() -> Result<()> {
	let mut nn = NeuralNet::new()?;
	train(&mut nn)
}


fn main() {
	if let Err(e) = run() {
		eprintln!("{}", e.to_string());
	}
}
