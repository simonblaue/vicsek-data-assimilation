
import animation
import vicsek

if __name__ =="__main__":
    anim = animation.VicsekAnimationConfig.exec_ref(animation.VicsekAnimationConfig,vicsek.OrderedSimulationConfig)
    anim(save_name=False)
# In gitignore
