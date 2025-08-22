use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use rust_ml::neural_network::NeuralNetwork;
use serde::{Deserialize, Serialize};

// Define a struct to hold API request data
#[derive(Serialize, Deserialize)]
struct RequestData {
    input: Vec<f64>,
}

// Define a struct to hold API response data
#[derive(Serialize, Deserialize)]
struct ResponseData {
    output: Vec<f64>,
}

async fn predict(req: web::Json<RequestData>) -> impl Responder {
    // Create a neural network instance
    let mut network = NeuralNetwork::new(vec![2, 3, 2]);

    // Set the input layer
    network.set_input_layer(req.input.clone());

    // Run the neural network
    let output = network.run();

    // Return the output as API response
    HttpResponse::Ok().json(ResponseData { output })
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .service(web::resource("/predict").route(web::post().to(predict)))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}