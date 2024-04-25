####################################################################################################
# This function converts any file into C/C++ source code.
# Example:
# - input file: data.dat
# - output file: data.h
# - variable name declared in output file: DATA
# - data length: sizeof(DATA)
# embed_resource("data.dat" "data.h" "DATA")
####################################################################################################

function(embed_resource_txt
	resource_file_name
       	source_file_name
       	variable_name)

    if(EXISTS "${source_file_name}")
        if("${source_file_name}" IS_NEWER_THAN "${resource_file_name}")
            return()
        endif()
    endif()

    file(STRINGS
            "${resource_file_name}"
            content_lines
            # ENCODING
    )
    set(content "")
    list(LENGTH content_lines list_size)
    math(EXPR list_max_index ${list_size}-1)
    foreach(index RANGE ${list_max_index})
        list(GET content_lines ${index} line_value)
        string(APPEND content "\"${line_value}\"\n")
    endforeach()

    set(array_definition
"#pragma once

static constexpr const char ${variable_name}[] =\n{\n${content}\n};")

    set(source "// Auto generated file.\n${array_definition}\n")

    file(WRITE "${source_file_name}" "${source}")

endfunction()

