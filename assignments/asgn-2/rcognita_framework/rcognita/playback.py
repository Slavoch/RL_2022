if np.size(ts) > 500:
    sim_step_size = int(np.floor((np.size(ts) / 500)))
    anim_interval = 1e-4
    Nframes = 500
else:
    sim_step_size = 1
    anim_interval = np.size(ts) / 1e6
    Nframes = np.size(ts)

sim_step_size = speedup * sim_step_size

# Down-sample and feed data into the animator
my_animator.get_sim_data(
    ts[:-sim_step_size:sim_step_size],
    xCoords[:-sim_step_size:sim_step_size],
    yCoords[:-sim_step_size:sim_step_size],
    alphas[:-sim_step_size:sim_step_size],
    vs[:-sim_step_size:sim_step_size],
    omegas[:-sim_step_size:sim_step_size],
    rs[:-sim_step_size:sim_step_size],
    icosts[:-sim_step_size:sim_step_size],
    Fs[:-sim_step_size:sim_step_size],
    Ms[:-sim_step_size:sim_step_size],
)

vids_folder = "vids"

if is_save_anim:
    ffmpeg_writer = animation.writers["ffmpeg"]
    metadata = dict(
        title="RL demo", artist="Matplotlib", comment="Robot parking example"
    )
    writer = ffmpeg_writer(fps=30, metadata=metadata)

# ------------------------------------main playback loop
# This is how you debug `FuncAnimation` if needed: just uncomment these two lines and comment out everything that has to do with `FuncAnimation`
# my_animator.init_anim()
# my_animator.animate(1)

anm = animation.FuncAnimation(
    my_animator.fig_sim,
    my_animator.animate,
    init_func=my_animator.init_anim,
    blit=False,
    interval=anim_interval,
    repeat=False,
    frames=Nframes,
)

cId = my_animator.fig_sim.canvas.mpl_connect(
    "key_press_event", lambda event: on_key_press(event, anm)
)

anm.running = True

my_animator.fig_sim.tight_layout()

plt.show()

if is_save_anim:
    anm.save(
        vids_folder + "/" + datafile.split(".")[0].split("/")[-1] + ".mp4",
        writer=writer,
        dpi=200,
    )

