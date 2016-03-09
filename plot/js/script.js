
// get label names
var methods = ["SIFT","SURF","BRISK","AKAZE","ORB"];
var measures = [
                  "Putative match ratio",
                  "Precision",
                  "Matching score",
                  "Recall",
                  "Entropy",
                  //"Number of keypoints 1",
                  "kpts1",
                  "kpts2",
                  "Correct matches",
                  "Correspondences",
                  "Tolerance"
               ];

var measures_to_display = [0,1,2,3,7,8];

var cont_width = 800;
var MAX_TIME = 4000;




function getTagContent(node,string)
{
  return node.getElementsByTagName(string)[0].innerHTML;
}

function getResults(test)
{
  return getTagContent(test,'data')
            .split(/\s+/) // remove spaces
            .filter(function(el) {return el.length != 0}); // remove empty string
}

function filterTestType(tests,testType)
{
  var out = [];

  for(var i=0; i<tests.length; i++)
  {
    var testType_ = getTagContent(tests[i],"alteration");
    if(testType_ == testType)
      out.push(tests[i]);
  }

  return out;
}


var container_transf = document.getElementById("results-transf"),
  container_noise = document.getElementById("results-noise"),
  container_time = document.getElementById("results-time");
// display_data('/data/res_transf_10.xml','transf',container_transf);
// display_data('/data/res_noise_10.xml','noise',container_noise);
display_data('/data/res.xml','transf',container_transf);
display_data('/data/res.xml','noise',container_noise);
display_time('/data/res.xml',container_time);


function display_data(filename, testType, container)
{
  var data, tests; 
  $.ajax({
      type: 'GET',
      url: filename,
      success: function (xml) {
          // process data
          data = xml;
          tests = data.getElementsByTagName('test');
          tests = filterTestType(tests,testType);

          // send to display functions
          var tables = setup_tables( container );
          fill_tables( tables,tests );
      }
  });
}


function setup_tables(container)
{
  var tables = [];
  for(var k=0; k<window.measures.length; k++)
  {
    var table = document.createElement("table"),
        thead = table.createTHead(),
        tbody = document.createElement("tbody");

    tables.push(table);

    // ---------------------------
    // fill table
    // ---------------------------

    for(var i=0; i<window.methods.length; i++)
    {
      // new row
      var r = table.insertRow(i);
      for(var j=0; j<window.methods.length; j++)
      {
        // new cell
        var c = r.insertCell(j);
        c.innerHTML = "-"//getResults();
      }
      // set 1st cell
      var r_label = r.insertCell(0);
      r_label.innerHTML = methods[i];
    }


    // ---------------------------
    // set table head
    // ---------------------------
    var first_row = thead.insertRow(0);
    for(var i=0; i<methods.length; i++)
    {
      var c = first_row.insertCell(i);
      c.innerHTML = methods[i];
    }
    first_row.insertCell(0);

    // ---------------------------
    // set caption
    // ---------------------------
    table.createCaption();
    var cap_name = document.createTextNode(measures[k]);
    table.caption.appendChild(cap_name);

    // ---------------------------
    // append table
    // ---------------------------
    container.appendChild(table);
  }

  return tables;
}

function fill_tables(tables,tests)
{
  for(var i=0; i<tests.length; i++)
  {
    var test = tests[i],
        x = methods.indexOf(getTagContent(test,'detector'))+1,
        y = methods.indexOf(getTagContent(test,'descriptor'))+1,
        results = getResults(test);

        var pc = true; 
        for(var j=0; j<results.length; j++)
        {
          if (j>3) pc=false;
          var numbers = parse_number(results[j],pc);
          tables[j].rows[x].cells[y].innerHTML = numbers[0];
          if (j<4)
          {
            tables[j].rows[x].cells[y].style["backgroundColor"] = colormap(numbers[1]);
          }
        }

  }
}


function parse_number(str,pc)
{
    // if it's a percentage
  if(pc)
  {
    var parts = str.split('e'),
      number,
      display_number;
    if(parts.length==1)
    {
      display_number = '100';
      number = 100;
    } 
    else 
    {
      number = Number(parts[0]*10);
      display_number = Math.round(number*100)/100;
    }

    return [String(display_number),number];
  }

  // else -> scientific notation
  var parts = str.split('e'),
    number = Number(parts[0]),
    display_number = Math.round(number*100)/100;

  if(parts.length == 2)
  {
    number_str = display_number + ' 10<sup>' + Number(parts[1]) + '</sup>'; 
    number *= Math.pow(10,parts[1]);
  } 
  else 
  {
    number_str = String(display_number);
  }

  return [number_str,number];
}

function colormap(n)
{
  return 'rgb(255,' + Math.round((100-(n-50)*3)*255/100) + ',0)';
}


function display_time(filename, container)
{
  var data, tests; 
  $.ajax({
      type: 'GET',
      url: filename,
      success: function (xml) {
          // process data
          data = xml;
          tests = data.getElementsByTagName('test');

          // send to display functions
          fill_times( container, tests );
      }
  });
}

function fill_times(container,tests)
{
  var done = [];

  // put legend
  var legend = add_times(container,"",[window.MAX_TIME/3.5,window.MAX_TIME/3.5,window.MAX_TIME/3.5]);
  legend[0].innerHTML = "detection";
  legend[1].innerHTML = "description";
  legend[2].innerHTML = "matching";

  for(var i=0; i<tests.length; i++ )
  {
    var detector = getTagContent(tests[i],'detector');
    var descriptor = getTagContent(tests[i],'descriptor');
    var hasTime = (tests[i].getElementsByTagName('times').length != 0 );
    var notDone =  (done.indexOf(detector)== -1);

    if(detector==descriptor && hasTime && notDone)
    {
      done.push(detector);
      var times = getTagContent(tests[i],'times').trim().split(' ');
      add_times(container,detector,times);
    }
  }
}

function get_px_width(element)
{
  return window.getComputedStyle(element)
               .width
               .split("px")[0];
}

function add_times(container,detector,times)
{
      var time_div = document.createElement("div"),
          title = document.createElement("h4");
      time_div.id = "time-cont-" + detector;
      time_div.className = "timebar";
      time_div.width = cont_width;
      title.innerHTML = detector;
      time_div.appendChild(title);
      var bars = [];


      for(var i=0; i<3; i++)
      {
        var time_part = document.createElement("div");
        time_part.id = "time-res-" + detector + "-" + i;
        time_part.className = "t" + i;
        time_part.style.width = Math.log(window.cont_width * times[i] / MAX_TIME)*30 +"px";
        time_part.innerHTML = "" + times[i] + " ms";// + "<br>" + 100*(times[i]/(times[0]+times[1]+times[2])) +" %";
        time_div.appendChild(time_part);
        bars.push(time_part);
      }
      container.appendChild(time_div);
      return bars;
}
