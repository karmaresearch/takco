// javascript: (function () { var d = document, s = d.createElement("script") s.setAttribute("src", "http://localhost:8000/embed.js") d.body.appendChild(s) })()

d = document
s = d.createElement("script")
s.setAttribute("src", "https://code.jquery.com/jquery-1.12.4.js")
d.body.appendChild(s)

function openBar() {
  document.getElementById('takcobar-content').style.marginTop = '35px'
  document.getElementById('takcobar-btn').innerHTML = '&nbsp;&#x25BC;&nbsp;🌮'
  document.getElementById('takcobar-btn').onclick = closeBar
}

function closeBar() {
  const neg = String($('#takcobar-content').height());
  document.getElementById('takcobar-content').style.marginTop = `-${neg}px`
  document.getElementById('takcobar-btn').innerHTML = '&nbsp;&#xA835;&nbsp;🌮'
  document.getElementById('takcobar-btn').onclick = openBar
  $("#takcobar-tabs > a").removeClass("tab-active")
  $('#takcobar-content').empty()
}

function showLoader() {
  const loader = document.getElementById('takcobar-loader')
  if (loader) { loader.style.display = 'inline' }
}
function hideLoader() {
  const loader = document.getElementById('takcobar-loader')
  if (loader) { loader.style.display = 'none' }
}

function getJSON(url, success, error) {
  showLoader()
  var request = new XMLHttpRequest()
  request.open('GET', url, true)
  request.onload = function() {
    hideLoader()
    if (this.status >= 200 && this.status < 400) {      
      success(JSON.parse(this.response))
    } else {
      if (error) { error() }
    }
  }
  request.onerror = function() {
    hideLoader()
    if (error) { error(error) }
  }
  request.send()
}

function postJSON(url, payload, success, error) {
  showLoader()
  var request = new XMLHttpRequest()
  request.open('POST', url, true)
  request.onload = function() {
    hideLoader()
    if (this.status >= 200 && this.status < 400) {      
      success(JSON.parse(this.response))
    } else {
      if (error) { error() }
    }
  }
  request.onerror = function() {
    hideLoader()
    if (error) { error(error) }
  }
  request.send(payload)
}
function populateForm(frm, data) {
  $.each(data, function(key, value){  
    var $ctrl = $('[name='+key+']', frm); 
    if($ctrl.is('select')){
      $("option",$ctrl).each(function(){
        if (this.value==value) { this.selected=true; }
      });
    }
    else {
      switch($ctrl.attr("type")) {  
          case "text" : case "hidden": case "textarea":
            $ctrl.val(value);   
            break;   
          case "radio" : case "checkbox":   
            $ctrl.each(function(){
              if($(this).attr('value') == value) {  $(this).attr("checked",value); } });   
            break;
      } 
    } 
  });  
}

function next() {
  window.curTableIndex++ ;
  if (window.curTableIndex == window.allTables.length) { window.curTableIndex = 0; }
  highlightTable(window.curTableIndex)
}
function prev() {
  if (window.curTableIndex  ==0 ) { window.curTableIndex = window.allTables.length; }
  window.curTableIndex-- ;
  highlightTable(window.curTableIndex)
}
function highlightTable() {
  closeBar()
  $('#curTableIndex').text(window.curTableIndex + 1)
  el = $(window.allTables[window.curTableIndex]);
  highlighted = $('.shadow-pulse').addClass('shadow-pulse-reverse')
  el.addClass('shadow-pulse');
  $('html, body').addClass("no-pointer-events").animate({
    scrollTop: el.offset().top - 200
  }, {
    duration: 500, 
    complete: function () {
      highlighted.removeClass('shadow-pulse').removeClass('shadow-pulse-reverse');
    }
  });
}

tabs = {
  "Reshape": () => {
    context = {
      "Page Title": '<a href="#">' + $('h1')[0].innerText + '</a>',
      "Header": $(el).prevAll(':header').first().children()[0].innerText,
    }
    rows = $.map(context, (v,k) => `<tr><th>${k}</th><td>${v}</td></tr>`).join('')
    $('#takcobar-content').append(
      $(`<div><h4>Context</h4><table>${rows}</table></div>`)
    )
    $('#takcobar-content').append(
      $(`<div>
      <h4>Heuristics</h4>
      <form>
        <input type="checkbox" />
        <select>
          <option>Regex</option>
        </select>
        <input type="text" value="foobar" />
      </form
      </div>`)
    )
    el = window.allTables[window.curTableIndex]
    payload = {
      outerHTML: el.outerHTML,
      classList: el.classList,
      context: context
    }
    postJSON('http://localhost:5000/reshape', JSON.stringify(payload), (data)=>{
      $('#takcobar-content').append(
        $(`<div></div>`).text(JSON.stringify(data.context))
      )
    }, (e)=>{
      console.error(e);
    })
  },
  "Cluster": () => {
    $('#takcobar-content').append(
      $(`<div><h4>Matchers</h4></div>`)
    )
  },
  "Link": () => {
    $('#takcobar-content').append(
      $(`<div><h4>Label Index</h4></div>`)
    )
    $('#takcobar-content').append(
      $(`<div><h4>KG</h4></div>`)
    )
  },
  "Extract": () => {
    $('#takcobar-content').append(
      $(`<div><table><tr><th>Entity</th><th>Statements</th></tr></table</div>`)
    )
  },
}


function addBar() {
  
  var sidebar = Object.assign(document.createElement('sidebar'), {
      id: 'takcobar',
      innerHTML: `
        <div id="takcobar-top">
          <a href="javascript:void(0)" id="takcobar-btn" onclick="closeBar()">&nbsp;</a>
          &nbsp;
          <input type="button" onclick="prev()" value="👈" />
          Table <span id="curTableIndex"></span> / <span id="totalTables"></span>
          <input type="button" onclick="next()" value="👉" />
          &nbsp;
          <span id="takcobar-tabs">
            <a href="javascript:void(0)">Reshape</a>
            <a href="javascript:void(0)">Cluster</a>
            <a href="javascript:void(0)">Link</a>
            <a href="javascript:void(0)">Extract</a>
          </span>
          <img id="takcobar-loader" src="http://localhost:5000/load.gif" height="100%"/>
        </div>
        <div id="takcobar-content">
          
        </div>   
      `,
  })

  window.allTables = $('.wikitable');
  if (window.allTables.length) {
    document.getElementsByTagName('body')[0].appendChild(sidebar)
    $("#takcobar-tabs > a").click((event) => {
      const e = $(event.target)
      if (e.hasClass("tab-active")) {
        closeBar()
      } else {
        $("#takcobar-tabs > a").removeClass("tab-active")
        $('#takcobar-content').empty()
        e.addClass("tab-active")
        openBar()
        tabs[e.text()]()
      }
    })
    window.curTableIndex = 0;
    $('#totalTables').text(window.allTables.length)
    $('#curTableIndex').text(window.curTableIndex + 1)
    highlightTable()
  }
}

bar = document.getElementById('takcobar')
if (bar) { bar.remove() }
var style = Object.assign(document.createElement('link'), {
  type: "text/css",
  rel: "stylesheet",
  href: "http://localhost:5000/bar.css?" + Math.random(), // prevent cache
  onload: addBar,
})
document.getElementsByTagName('head')[0].appendChild(style)

