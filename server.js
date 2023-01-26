var http = require('http');
var fs = require('fs');
var path = require("path");
const express = require('express')
const app = express()
const bodyParser = require("body-parser");


app.use(bodyParser.json({
	limit: '50mb'
  }));
  
  app.use(bodyParser.urlencoded({
	limit: '50mb',
	parameterLimit: 100000,
	extended: true 
  }));

app.get('/', (req, res) => {
	res.sendFile(path.join(__dirname + '/fire.html'))
})

app.post('/checkfire',async(req, res) =>{
	image = req.body.image
	const axios = require('axios')

	axios
	.post('http://127.0.0.1:7778//checkingfire', {image: image})
	.then(mlresult => {
		console.log(mlresult["data"])
		if (mlresult["data"] == "Fire"){
			res.json({ status: true })
		}
		else{
			res.json({ status: false })
		}
	})

})
app.listen(7777, () => {
	console.log('Server up at 7777')
})