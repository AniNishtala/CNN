# Working directories.
set project_root "C:/Xilinx/CMPE240/Assignment5"
set build_root "$project_root/tcl_build_single_batch"

# Project settings.
set fpga_part "xc7k160tfbg676-2"
set clock_period_ns 10

# Run both components.
set components {Lab5 ExtraCredit}

# 1 to run, 0 to skip.
set run_csim 1
set run_csynth 1
set run_cosim 1
set run_impl 1

# 1 checks files without creating components.
set dry_run 0

# All bonus images in ONE RTL simulation.
set bonus_images 10000

# Disable large waveform files.
set cosim_trace_level none

# Required files.
set design_files {
    cnn.cpp
    header.h
    weights.h
}

set lab5_data {
    image.dat
    trace.dat
}

set bonus_data {
    test_count.dat
    test_images.dat
    test_labels.dat
    keras_labels.dat
    keras_probabilities.dat
}

# Check all files before creating components.
foreach component $components {
    set source_dir "$project_root/$component"

    if {$component == "Lab5"} {
        set data_files $lab5_data
    } elseif {$component == "ExtraCredit"} {
        set data_files $bonus_data
    } else {
        error "Unknown component: $component"
    }

    set required_files [concat $design_files {tb_cnn.cpp} $data_files]

    foreach filename $required_files {
        if {![file isfile "$source_dir/$filename"]} {
            error "Missing file: $source_dir/$filename"
        }
    }

    if {$component == "ExtraCredit"} {
        set f [open "$source_dir/test_count.dat" r]
        set exported_images [string trim [read $f]]
        close $f

        if {$exported_images != $bonus_images} {
            error "Expected $bonus_images images, found $exported_images."
        }
    }
}

puts "All required files were found."

if {$dry_run == 1} {
    puts "File check completed. No components were created."
    exit 0
}

if {$run_csynth == 0 && ($run_cosim == 1 || $run_impl == 1)} {
    error "Set run_csynth to 1 for co-simulation or implementation."
}

file mkdir $build_root

foreach component $components {
    set source_dir "$project_root/$component"

    if {$component == "Lab5"} {
        set component_name cnn_part3
        set top_function GemmConv2d0
        set data_files $lab5_data
    } else {
        set component_name cnn_extra_credit
        set top_function CNN
        set data_files $bonus_data
    }

    puts "Starting $component"
    flush stdout

    cd $build_root

    # Reset only this generated component.
    # Original source folders and other build folders are preserved.
    open_component -reset -flow_target vivado $component_name
    set_top $top_function

    # Design source files.
    foreach filename $design_files {
        add_files "$source_dir/$filename"
    }

    # Testbench and its data.
    add_files -tb "$source_dir/tb_cnn.cpp"

    foreach filename $data_files {
        add_files -tb "$source_dir/$filename"
    }

    set_part $fpga_part
    create_clock -period $clock_period_ns -name default

    if {$run_csim == 1} {
        cd $build_root

        puts "$component: starting C simulation"
        flush stdout

        # No arguments: full dataset for the bonus testbench.
        csim_design -clean

        puts "$component: C simulation completed"
        flush stdout
    }

    if {$run_csynth == 1} {
        cd $build_root

        puts "$component: starting HLS synthesis"
        flush stdout

        csynth_design

        puts "$component: HLS synthesis completed"
        flush stdout
    }

    if {$run_cosim == 1} {
        cd $build_root

        puts "$component: starting RTL co-simulation"
        flush stdout

        if {$component == "ExtraCredit"} {
            puts "Testing all $bonus_images images in ONE RTL batch."
            flush stdout

            # Compatible with both the original count-only testbench
            # and the updated count/offset testbench.
            # Starting image defaults to zero.
            cosim_design -rtl verilog \
                -trace_level $cosim_trace_level \
                -argv $bonus_images
        } else {
            cosim_design -rtl verilog \
                -trace_level $cosim_trace_level
        }

        puts "$component: RTL co-simulation completed"
        flush stdout
    }

    if {$run_impl == 1} {
        cd $build_root

        puts "$component: starting Vivado implementation"
        flush stdout

        export_design -flow impl \
            -rtl verilog \
            -format ip_catalog

        puts "$component: implementation completed"
        flush stdout
    }

    if {[llength [info commands close_component]] > 0} {
        close_component
    }

    puts "$component completed."
    flush stdout
}

puts "All selected components completed."
puts "Generated projects and reports: $build_root"
flush stdout

exit 0