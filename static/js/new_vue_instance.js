function VueInstance(aI){
new Vue({
    el: '#app',
        data() {
        return {
            navList:[
                {name:'/index',navItem:'首页'},
                {name:'/trainCenter',navItem:'训练中心'},
                {name:'/predictData',navItem:'数据预测'},
                {name:'/about',navItem:'关于'}],
            activeIndex:aI
            }
        },
    methods: {
      handleSelect(key, keyPath) {
        console.log(key, keyPath);
        },
        handleCommand(command){  // 用户名下拉菜单选择事件
      if(command == 'logout'){
        window.location.replace("/do_logout/");
      }else if(command=='changePass'){
        window.location.replace("/password_change/");
      }
    }
   }
})
}