@task
def runGrid(c, full_cmd, autoname=True, detectron=False, test=False, sleep=False, gpu_nb=1.0, interactive=False, portforward=False, big_gpu=False, h100=False, save_log=False, lname="test_run", comments="", memory="50G", nbcpu=12, rcp=False):
    if detectron:
        full_cmd = full_cmd + " OUTPUT_DIR /cvlabdata2/home/engilber/dev/domain_adaptation/MNIST-ObjectDetection/all_outs/{automodelname}"
    elif autoname and not test:
        full_cmd = full_cmd + " -n {automodelname}"

    if test:
        full_cmd = "python " + full_cmd
    else:
        git_commit, model_name, full_cmd = sh.expe("log", "-g", "--comment", comments, ("python " + full_cmd).split()).split("\n")[:3]
        print(git_commit)
        print(model_name)
        print(full_cmd)
        
        full_cmd = eval(full_cmd)
        full_cmd = " ".join(full_cmd).format(automodelname=model_name)

    print(full_cmd)

    base_job_name, image_name, workdir, cmd_setup_env, scratch_dir = load_local_grid_config()
    job_name = get_next_runai_job_name(base_job_name)

    full_cmd = full_cmd


    # git clone {workdir} /home/engilber/project_dir
    # cd /home/engilber/project_dir
    # git checkout {git_commit}

    # cd /home/engilber/project_dir

    if scratch_dir is not None:
        # check if scratch dir exists
        scratch_dir_exists = bool(int(sh.ssh(shlex.split(f"engilber@iccvlabsrv15 'if test -d /cvlabscratch{scratch_dir}; then echo 1; else echo 0; fi;'"))))
        if not scratch_dir_exists:
            # if not clone it from workdir
            scratch_clone_cmd = f'engilber@iccvlabsrv15 "git clone {workdir} /cvlabscratch{scratch_dir}"'
        else:
            # else pull it
            scratch_clone_cmd = f'engilber@iccvlabsrv15 "cd /cvlabscratch{scratch_dir}; git pull"'    
        
        sh.ssh(shlex.split(scratch_clone_cmd))

        workdir = "/cvlabscratch/cvlab" + scratch_dir


    if sleep:
        setup_script = "/opt/lab/setup_and_wait.sh"
    elif detectron:
        setup_script = "/opt/lab/setup_and_run_command.sh"
    else:
        setup_script = "bash -c"

    if interactive:
        interactive_arg = "--interactive"
    else:
        interactive_arg = ""

    if memory is not None:
        memory_arg = f"--memory {memory} --memory-limit {memory}"
    else:
        memory_arg = ""

    if nbcpu is not None:
        processor_arg = f"--cpu {nbcpu} --cpu-limit {nbcpu}"
    else:
        processor_arg = ""


    if big_gpu:
        if interactive:
            node_type = "--node-pools g10 --preemptible"
        else:
            node_type = "--node-pools g10"

    if h100:
        node_type = "--node-pools h100 default"
        
    else:
        node_type = "--node-pools default"

    if interactive and portforward:
        interactive_arg = '--interactive --service-type portforward --address "0.0.0.0" --port 63499:8888'

    if interactive:
        command = f"{setup_script}"
    elif test:
        # command = f"{setup_script} \"cd {workdir} && {cmd_setup_env} && cd {workdir} && {full_cmd}\"" #| tee -a {workdir+'/logs/'+lname+'_log.txt'}
        command = f"{setup_script} \"git clone {workdir} /home/engilber/project_dir && cd /home/engilber/project_dir && {cmd_setup_env} && cd /home/engilber/project_dir && {full_cmd}\"" # |&| tee -a {workdir+'/logs/'+model_name+'_log.log'}

    else:
        command = f"{setup_script} \"git clone {workdir} /home/engilber/project_dir && cd /home/engilber/project_dir &&  git checkout {git_commit} && {cmd_setup_env} && cd /home/engilber/project_dir && {full_cmd}\"" # |&| tee -a {workdir+'/logs/'+model_name+'_log.log'}

    if save_log:
        log_filename = lname if test else model_name
        command = command + f" |& tee -a {workdir+'/logs/'+log_filename+'_log.log'}"

    pvc_arg = "--pvc cvlab-scratch:/cvlabscratch" if rcp else "--pvc runai-cvlab-engilber-scratch:/cvlabscratch"

    grid_command = f"runai submit {job_name} \
            -i {image_name} \
            --gpu {gpu_nb} \
            {pvc_arg} \
            --large-shm \
            --allow-privilege-escalation i\
            {interactive_arg} \
            {memory_arg} \
            {processor_arg} \
            -e CLUSTER_USER=engilber \
            -e CLUSTER_USER_ID=113790 \
            -e CLUSTER_GROUP_NAME=CVLAB-unit \
            -e CLUSTER_GROUP_ID=11166 \
            -e AUTO_SHUTDOWN_TIME=3h \
            -e MPLBACKEND=Agg \
            {node_type} \
            --command -- {command}"
            # --pvc runai-cvlab-engilber-cvlabdata2:/cvlabdata2 \
            # --pvc runai-cvlab-engilber-cvlabsrc1:/cvlabsrc1 \
            # --pvc runai-pv-cvlabdata1:/cvlabdata1 \
    print(grid_command)
    # runai-cvlab-scratch-engilber:/cvlabscratch
    # runai-cvlab-engilber-scratch:/cvlabscratch
    
    c.run(grid_command)




"""
runai submit --name test -i registry.rcp.epfl.ch/cvlab-unit-grosche/blur:v0.1 --gpu 1 --large-shm -e LDAP_USERNAME=grosche -e LDAP_UID=260305 -e LDAP_GROUPNAME=cvlab-unit -e LDAP_GID=11166 --pvc cvlab-scratch:/cvlabscratch --interactive --command -- "pip install onnxruntime-gpu==1.21.0 /bin/bash -ic "sleep 3600"


runai submit --name test -i registry.rcp.epfl.ch/cvlab-unit-grosche/blur:v0.1 --gpu 1 --large-shm -e LDAP_USERNAME=grosche -e LDAP_UID=260305 -e LDAP_GROUPNAME=cvlab-unit -e LDAP_GID=11166 --pvc cvlab-scratch:/cvlabscratch --command -- pip install onnxruntime && python blur_faces.py


runai submit \
  --name test \
  --image registry.rcp.epfl.ch/cvlab-unit-grosche/blur:v0.1 \
  --gpu 1 \
  --large-shm \
  -e LDAP_USERNAME=grosche \
  -e LDAP_UID=260305 \
  -e LDAP_GROUPNAME=cvlab-unit \
  -e LDAP_GID=11166 \
  --pvc cvlab-scratch:/cvlabscratch \
  --interactive \
  --command -- /bin/bash -ic "pip install onnxruntime-gpu --upgrade && sleep 3600"
"""