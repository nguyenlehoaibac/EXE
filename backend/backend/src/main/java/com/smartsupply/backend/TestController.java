package com.smartsupply.backend;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import java.util.List;

@RestController
public class TestController {

    @Autowired
    private CalendarService calendarService;

    @GetMapping("/api/test")
    public String sayHello() {
        return "Xin chao! May chu SmartSupplyAI Backend dang hoat dong rat tot!";
    }

    @GetMapping("/api/calendar")
    public List<CalendarDTO> getCalendar() {
        // Trả về danh sách đối tượng, Spring Boot sẽ tự động chuyển thành JSON chuẩn mực
        return calendarService.readCalendarCsv();
    }
}