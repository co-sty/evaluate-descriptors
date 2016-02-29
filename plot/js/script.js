
var data = jsyaml.load( d3.select('#results').text() );


var height = 200,
    width = height;

svg = d3.select('body')
//	.style('overflow-x','hidden')
	.append('svg')
	.attr('height', height)
	.attr('width', width);

g = svg.append('g')
	.attr('transform','translate('+(-1*width/2)+')');      
			
pattern = g.append('defs') 
	.selectAll('.pattern')
	.data( data['item'] )
