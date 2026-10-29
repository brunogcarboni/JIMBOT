# Accessing the FGPA and using ARM as a clock-source

1. Within the Vivado directory, open a terminal and type `xsct` and press `Enter`.

2. Then:
    ```shell
    connect
    targets -set -filter {name =^ "ARM*#0"}
    rst -processor
    ```
    
3. At last, go to `vivado_projects/{vivados_prj_name}.gen/sources_1/bd/zynq_processing_system7_0_0/` and open a terminal (in this folder), then write this:

    ```shell
    source ps7_init.tcl
    ps7_init
    ps7_post_config
    ```
These commands reset the ARM core inside the Zynq SoC and prepare it to be used by the user-space programs.