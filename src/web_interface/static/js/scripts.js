// This function is called when the page is loaded.
function init() { 
    revealTitle();
}

// This function is called when a user clicks the browse button to upload a file.
function browse() {
    // Get the file input element.
    var file_input = document.getElementById('file_input');
    // Click the file input element.
    file_input.click();
}

// This function is called when a user selects a file to upload.
function file_selected() {
    // Get the file input element.
    var file_input = document.getElementById('file_input');
    // Get the file name element.
    var file_name = document.getElementById('file_name');
    // Set the file name element to the name of the file.
    file_name.innerHTML = file_input.files[0].name;
}

// This function is called when a user clicks the upload button.
function upload() { 
    // Get the file input element.
    var file_input = document.getElementById('file_input');
    // Get the file name element.
    var file_name = document.getElementById('file_name');
    // Get the file name.
    var file = file_input.files[0].name;
    // Check if the file name is empty.
    if (file === '') {
        // Get the error message element.
        var error_message = document.getElementById('error_message');
        // Set the error message element to the error message.
        error_message.innerHTML = 'Please select a file.';
        // Show the error message element.
        error_message.style.display = 'block';
        return;
    }
    // Get the file extension.
    var extension = file.split('.').pop();
    // Check if the file extension is jpg or jpeg.
    if (extension.toLowerCase() === 'jpg' || extension.toLowerCase() === 'jpeg') {
        // Get the form element.
        var form = document.getElementById('form');
        // Submit the form.
        form.submit();
    } else {
        // Get the error message element.
        var error_message = document.getElementById('error_message');
        // Set the error message element to the error message.
        error_message.innerHTML = 'Please upload a jpg or jpeg.';
        // Show the error message element.
        error_message.style.display = 'block';
    }
}

// This function will bring in the title in a unique way.
function revealTitle() {
    // Get the title element.
    var title = document.getElementById('title');
    // Get the title text.
    var title_text = title.innerHTML;
    // Set the title element to an empty string.
    title.innerHTML = '';
    // Loop through each letter in the title text.
    for (var i = 0; i < title_text.length; i++) {
        // Create a span element.
        var span = document.createElement('span');
        // Set the span element to the current letter.
        span.innerHTML = title_text[i];
        // Set the span element to the current letter.
        span.style.animationDelay = i * 0.1 + 's';
        // Add the span element to the title element.
        title.appendChild(span);
    }
}



// Call the init function when the page is loaded.
document.addEventListener('DOMContentLoaded', init);
