// Iris dataset NN
use rand::prelude::SliceRandom;
use rand::distributions::Distribution;
use statrs::distribution::Normal;

const FILENAME: &'static str = "Iris.csv";
const TEST_DATA: &'static str = "iris_test.csv";
const NORM_CONST: f32 = 10.0;

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
	sepal_length: f32,
	sepal_width: f32,
	petal_length: f32,
	petal_width: f32,
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
				0 => {},
				1 => s.sepal_length = f.parse::<f32>()? / NORM_CONST,
				2 => s.sepal_width = f.parse::<f32>()? / NORM_CONST,
				3 => s.petal_length = f.parse::<f32>()? / NORM_CONST,
				4 => s.petal_width = f.parse::<f32>()? / NORM_CONST,
				5 => s.species = Species::from_string(f)?,
				_ => panic!("bad csv data")
			}
		}
		samples.push(s);
	}

	Ok(samples)
}

struct Weights {
	// Our first layer has 16 nodes
	// Each taking 4 inputs and a relu
	input_layer: [[f32; 4]; 16],
	input_layer_bias: [f32; 16],

	// Our output layer has 3 nodes, each taking
	// 16 inputs. We then apply a softmax.
	output_layer: [[f32; 16]; 3],
	output_layer_bias: [f32; 3],
}

struct NeuralNet {
	weights: Weights,

	// Store input
	input: Sample,
	input_comp: [f32; 16],
	input_comp_relu: [f32; 16],

	// Store output
	output_comp: [f32; 3],
	output_comp_softmax: [f32; 3],
}

impl NeuralNet {
	fn new() -> Result<Self> {

		let mut weights = Weights{
				input_layer: [[0.0; 4]; 16],
				input_layer_bias: [0.0; 16],
				output_layer: [[0.0; 16]; 3],
				output_layer_bias: [0.0; 3],
			};
		let mut t_normal = TruncatedNormal::new(0.0, 0.5)?;
		weights.input_layer.iter_mut().flatten()
			.chain(weights.output_layer.iter_mut().flatten())
			.chain(weights.input_layer_bias.iter_mut())
			.chain(weights.output_layer_bias.iter_mut())
			.for_each(|w| *w = t_normal.sample());

		Ok(NeuralNet{
			weights: weights,
			input: Sample::default(),
			input_comp: [0.0; 16],
			input_comp_relu: [0.0; 16],
			output_comp: [0.0; 3],
			output_comp_softmax: [0.0; 3],
		})
	}

	fn compute(&mut self, sample: Sample) -> [f32; 3] {
		// Save the sample
		self.input = sample;

		// For each of the 16 nodes
		for (node_num, node) in self.weights.input_layer.iter().enumerate() {
			// Each node is [f32; 4]
			// Perform a weighted sum
			let mut sum = 0.0;
			for (n, w) in node.iter().enumerate() {
				sum += match n {
					0 => w * sample.sepal_length,
					1 => w * sample.sepal_width,
					2 => w * sample.petal_length,
					3 => w * sample.petal_width,
					_ => panic!("too many weights")
				};
			}

			// Store the sum in input_comp
			self.input_comp[node_num] = sum 
				+ self.weights.input_layer_bias[node_num];

			// Store the relu value
			self.input_comp_relu[node_num] = relu(sum);
		}
		self.compute_out()
	}

	fn compute_out(&mut self) -> [f32; 3] {
		// Iterate over our three output nodes
		for (node_num, node) in self.weights.output_layer.iter().enumerate() {
			let mut sum = 0.0;
			for (w, v) in node.iter().zip(self.input_comp_relu.iter()) {
				sum += w * v;
			}

			// Set this
			self.output_comp[node_num] = sum
				+ self.weights.output_layer_bias[node_num];
		}

		self.output_comp_softmax = softmax(self.output_comp);
		self.output_comp_softmax
	}

	fn train(&mut self, learning_rate: f64) {
		let ind = match self.input.species {
			Species::Setosa => 0,
			Species::Versicolor => 1,
			Species::Virginica => 2,
			_ => panic!("unknown type not allowed"),
		};

		let d_loss_d_out = -1.0 / self.output_comp_softmax[ind];
		let mut d_loss_d_t = [d_loss_d_out as f64; 3];
		let sum_exp = [
			(self.output_comp[0] as f64).exp(),
			(self.output_comp[1] as f64).exp(),
			(self.output_comp[2] as f64).exp(),
		].iter().sum::<f64>();
		let class_exp = (self.output_comp[ind] as f64).exp();

		for (num, w) in d_loss_d_t.iter_mut().enumerate() {
			if num == ind {
				*w *= (class_exp * (sum_exp - class_exp)) 
					/ (sum_exp.powf(2.0));
			} else {
				let exp = (self.output_comp[num] as f64).exp();
				*w *= (-1.0 * exp * class_exp) / (sum_exp.powf(2.0));
			}
		}

		// From d_loss_d_t we need to compute
		//   - d_loss_d_b
		//   - d_loss_d_w
		//   - d_loss_d_i

		// d_t_d_b is 1 in all cases
		for (&w, b) in d_loss_d_t.iter()
			.zip(self.weights.output_layer_bias.iter_mut()) {

			*b -= (w * learning_rate) as f32;
		}

		// d_loss_d_w considers 16 weights for each node
		for (&t, node) in d_loss_d_t.iter()
			.zip(self.weights.output_layer.iter_mut()) {
			// We need to consider each of the 16 weights

			for (w, &v) in node.iter_mut()
				.zip(self.input_comp_relu.iter()) {

				*w -= (t * v as f64 * learning_rate) as f32;
			}
		}

		// d_loss_d_i
		let mut d_loss_d_i = [0.0 as f64; 16];
		for (ind, i) in d_loss_d_i.iter_mut().enumerate() {
			*i = [
				self.weights.output_layer[0][ind] as f64 * d_loss_d_t[0],
				self.weights.output_layer[1][ind] as f64 * d_loss_d_t[1],
				self.weights.output_layer[2][ind] as f64 * d_loss_d_t[2],
			].iter().sum::<f64>()
			* self.input_comp_relu[ind] as f64;
		}

		// d_loss_d_t means going through the relu
		let mut d_loss_d_t = d_loss_d_i.clone();
		for (i, &v) in d_loss_d_t.iter_mut().zip(self.input_comp.iter()) {
			*i = if v > 0.0 {
				*i
			} else {
				0.0
			};
		}

		// Update the bias
		for (b, &w) in self.weights.input_layer_bias.iter_mut()
			.zip(d_loss_d_t.iter()) {

			*b -= (learning_rate * w) as f32;
		}

		// Update the weights - each node has four weights
		for (node_num, node) in self.weights.input_layer.iter_mut()
			.enumerate() {

			for (n, w) in node.iter_mut().enumerate() {
				*w -= learning_rate as f32
					* d_loss_d_t[node_num] as f32
					* match n {
						0 => self.input.sepal_width,
						1 => self.input.sepal_length,
						2 => self.input.petal_width,
						3 => self.input.petal_length,
						_ => panic!("more than four iteration"),
					};
			}
		}
	}
}

fn train(nn: &mut NeuralNet) -> Result<()> {
	let mut samples = read_samples(FILENAME)?;
	let mut rng = rand::thread_rng();

	// Read / Create NN
	for _ in 0..2000 {
		samples.shuffle(&mut rng);

		for &s in samples.iter() {
			// Train with learning_rate = 0.001
			nn.compute(s);
			nn.train(0.001);
		}
	}

	Ok(())
}

fn test(nn: &mut NeuralNet) -> Result<()> {
	let samples = read_samples(TEST_DATA)?;

	// Read / Create NN
	let mut total_loss = 0.0;

	for &s in samples.iter() {
		print!("species = {:?}\t", s.species);
		let ans = nn.compute(s);
		print!("[{:.2}, {:.2}, {:.2}]\t", ans[0], ans[1], ans[2]);
		let loss = -1.0 * match s.species {
			Species::Setosa => ans[0].ln(),
			Species::Versicolor => ans[1].ln(),
			Species::Virginica => ans[2].ln(),
			_ => panic!("unknown species"),

		};
		println!("loss = {}", loss);
		total_loss += loss;
	}
	println!("\ntotal loss = {}", total_loss);

	Ok(())
}

fn run() -> Result<()> {

	let mut nn = NeuralNet::new()?;
	train(&mut nn)?;
	println!();
	println!("tests");

	test(&mut nn)
}


fn main() {
	if let Err(e) = run() {
		eprintln!("{}", e.to_string());
	}

}

struct TruncatedNormal {
	norm: statrs::distribution::Normal,
	rng: rand::rngs::ThreadRng,
	min: f32,
	max: f32,
}

impl TruncatedNormal {
	fn new(mean: f64, sd: f64) -> Result<Self> {
		let r = rand::thread_rng();
		let n = Normal::new(mean, sd)?;
		Ok(TruncatedNormal{
			norm: n,
			rng: r,
			min: (mean - 2.0 * sd) as f32,
			max: (mean + 2.0 * sd) as f32,
		})
	}

	fn sample(&mut self) -> f32 {
		loop {
			let s = self.norm.sample(&mut self.rng) as f32;
			if s < self.max && s > self.min {
				return s;
			}
		}
	}
}

fn softmax(vals: [f32; 3]) -> [f32; 3] {
	let mut exps = [(vals[0] as f64).exp(),
		        	(vals[1] as f64).exp(),
			    	(vals[2] as f64).exp()];
	let sum = exps.iter().sum::<f64>();
	exps[0] /= sum;
	exps[1] /= sum;
	exps[2] /= sum;

	[exps[0] as f32, exps[1] as f32, exps[2] as f32]
}

fn relu(val: f32) -> f32 {
	if val > 0.0 {val} else {0.0}
}